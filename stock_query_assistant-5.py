import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

system_prompt = f"""我是股票查询助手，以下是关于股票历史价格表 stock_price 的字段，我可能会编写对应的SQL，对数据进行查询

**当前系统时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

-- 股票历史价格表
CREATE TABLE stock_price (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    stock_name VARCHAR(20) NOT NULL COMMENT '股票名称',
    ts_code VARCHAR(20) NOT NULL COMMENT '股票代码',
    trade_date VARCHAR(10) NOT NULL COMMENT '交易日期',
    open DECIMAL(15,2) COMMENT '开盘价',
    high DECIMAL(15,2) COMMENT '最高价',
    low DECIMAL(15,2) COMMENT '最低价',
    close DECIMAL(15,2) COMMENT '收盘价',
    vol DECIMAL(20,2) COMMENT '成交量',
    amount DECIMAL(20,2) COMMENT '成交额',
    UNIQUE KEY uniq_stock_date (ts_code, trade_date)
);
我将回答用户关于股票历史价格的相关问题。
每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
如果是预测未来价格，需要对未来的价格进行详细的解释说明，比如价格将持续走高，或价格将相对平稳，或价格将持续走低。
"""

functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                },
                "need_visualize": {
                    "type": "boolean",
                    "description": "是否需要可视化和统计信息，默认True。如果是对比分析等场景可设为False，不进行可视化。",
                    "default": True
                }
            },
            "required": ["sql_input"],
        },
    },
    {
        "name": "arima_stock",
        "description": "对指定股票(ts_code)的收盘价进行ARIMA(5,1,5)建模，并预测未来n天的价格，返回预测表格和折线图。",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填"
                },
                "n": {
                    "type": "integer",
                    "description": "预测未来天数，必填"
                }
            },
            "required": ["ts_code", "n"]
        }
    },
    {
        "name": "boll_detection",
        "description": "使用布林带(20日周期+2σ)检测股票的异常点，识别超买和超卖日期。",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填"
                },
                "start_date": {
                    "type": "string",
                    "description": "开始日期(YYYY-MM-DD格式)，可选，默认一年前"
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期(YYYY-MM-DD格式)，可选，默认今天"
                }
            },
            "required": ["ts_code"]
        }
    },
    {
        "name": "prophet_analysis",
        "description": "使用Prophet模型对股票进行周期性分析，分析trend、weekly、yearly趋势，并可视化展示。",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填"
                },
                "start_date": {
                    "type": "string",
                    "description": "开始日期(YYYY-MM-DD格式)，可选，默认一年前"
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期(YYYY-MM-DD格式)，可选，默认今天"
                }
            },
            "required": ["ts_code"]
        }
    }
]

@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [
        {
            'name': 'sql_input',
            'type': 'string',
            'description': '生成的SQL语句',
            'required': True
        },
        {
            'name': 'need_visualize',
            'type': 'boolean',
            'description': '是否需要可视化和统计信息，默认True。如果是对比分析等场景可设为False，不进行可视化。',
            'required': False,
            'default': True
        }
    ]
    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        args = json.loads(params)
        sql_input = args['sql_input']
        database = args.get('database', 'stock')
        engine = create_engine(
            f"mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/{database}?charset=utf8mb4",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        try:
            df = pd.read_sql(sql_input, engine)
            # 前5行+后5行拼接展示
            if len(df) > 10:
                md = pd.concat([df.head(5), df.tail(5)]).to_markdown(index=False)
            else:
                md = df.to_markdown(index=False)
            # 只返回表格
            if len(df) == 1:
                return md
            need_visualize = args.get('need_visualize', True)
            if not need_visualize:
                return md
            desc_md = df.describe().to_markdown()
            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'stock_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            # 智能选择可视化方式
            generate_smart_chart_png(df, save_path)
            img_path = os.path.join('image_show', filename)
            img_md = f'![图表]({img_path})'
            return f"{md}\n\n{desc_md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

def generate_smart_chart_png(df_sql, save_path):
    columns = df_sql.columns
    if len(df_sql) == 0 or len(columns) < 2:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, '无可视化数据', ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        return
    x_col = columns[0]
    y_cols = columns[1:]
    x = df_sql[x_col]
    # 如果数据点较多，自动采样10个点
    if len(df_sql) > 20:
        idx = np.linspace(0, len(df_sql) - 1, 10, dtype=int)
        x = x.iloc[idx]
        df_plot = df_sql.iloc[idx]
        chart_type = 'line'
    else:
        df_plot = df_sql
        chart_type = 'bar'
    plt.figure(figsize=(10, 6))
    for y_col in y_cols:
        if chart_type == 'bar':
            plt.bar(df_plot[x_col], df_plot[y_col], label=str(y_col))
        else:
            plt.plot(df_plot[x_col], df_plot[y_col], marker='o', label=str(y_col))
    plt.xlabel(x_col)
    plt.ylabel('数值')
    plt.title('股票数据统计')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@register_tool('arima_stock')
class ArimaStockTool(BaseTool):
    description = '对指定股票(ts_code)的收盘价进行ARIMA(5,1,5)建模，并预测未来n天的价格，返回预测表格和折线图。'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填',
            'required': True
        },
        {
            'name': 'n',
            'type': 'integer',
            'description': '预测未来天数，必填',
            'required': True
        }
    ]
    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        ts_code = args['ts_code']
        n = int(args['n'])
        # 获取今天和一年前的日期
        today = datetime.now().date()
        start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        # 连接MySQL，获取历史收盘价
        engine = create_engine(
            f"mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/stock?charset=utf8mb4",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        sql = f"""
            SELECT trade_date, close FROM stock_price
            WHERE ts_code = '{ts_code}' AND trade_date >= '{start_date}' AND trade_date < '{end_date}'
            ORDER BY trade_date ASC
        """
        df = pd.read_sql(sql, engine)
        if len(df) < 30:
            return '历史数据不足，无法进行ARIMA建模预测。'
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        # ARIMA建模
        try:
            model = ARIMA(df['close'], order=(5,1,5))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=n)
            # 生成预测日期
            last_date = pd.to_datetime(df['trade_date'].iloc[-1])
            pred_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(n)]
            pred_df = pd.DataFrame({'预测日期': pred_dates, '预测收盘价': forecast})
            # 保存预测图
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'arima_{ts_code}_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            plt.figure(figsize=(10,6))
            plt.plot(df['trade_date'], df['close'], label='历史收盘价')
            plt.plot(pred_df['预测日期'], pred_df['预测收盘价'], marker='o', label='预测收盘价')
            plt.xlabel('日期')
            plt.ylabel('收盘价')
            plt.title(f'{ts_code} 收盘价ARIMA预测')
            plt.legend()
            # 横坐标自动稀疏显示
            all_dates = list(df['trade_date']) + list(pred_df['预测日期'])
            total_len = len(all_dates)
            if total_len > 12:
                step = max(1, total_len // 10)
                show_idx = list(range(0, total_len, step))
                show_labels = [all_dates[i] for i in show_idx]
                plt.xticks(show_idx, show_labels, rotation=45)
            else:
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            img_path = os.path.join('image_show', filename)
            img_md = f'![ARIMA预测]({img_path})'
            return f"{pred_df.to_markdown(index=False)}\n\n{img_md}"
        except Exception as e:
            return f'ARIMA建模或预测出错: {str(e)}'

@register_tool('boll_detection')
class BollDetectionTool(BaseTool):
    description = '使用布林带(20日周期+2σ)检测股票的异常点，识别超买和超卖日期。'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填',
            'required': True
        },
        {
            'name': 'start_date',
            'type': 'string',
            'description': '开始日期(YYYY-MM-DD格式)，可选，默认一年前',
            'required': False
        },
        {
            'name': 'end_date',
            'type': 'string',
            'description': '结束日期(YYYY-MM-DD格式)，可选，默认今天',
            'required': False
        }
    ]
    
    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        ts_code = args['ts_code']
        start_date = args.get('start_date')
        end_date = args.get('end_date')
        
        # 如果没有指定日期，默认使用过去一年
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 连接MySQL，获取历史收盘价
        engine = create_engine(
            f"mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/stock?charset=utf8mb4",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        
        sql = f"""
            SELECT trade_date, close FROM stock_price
            WHERE ts_code = '{ts_code}' AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date ASC
        """
        
        try:
            df = pd.read_sql(sql, engine)
            if len(df) < 30:
                return f'历史数据不足，至少需要30个交易日的数据进行布林带分析。当前只有{len(df)}条数据。'
            
            # 数据预处理
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 计算布林带指标 (20日周期 + 2σ)
            window = 20
            df['MA20'] = df['close'].rolling(window=window).mean()
            df['STD20'] = df['close'].rolling(window=window).std()
            df['Upper_Band'] = df['MA20'] + 2 * df['STD20']
            df['Lower_Band'] = df['MA20'] - 2 * df['STD20']
            
            # 检测异常点
            df['超买'] = df['close'] > df['Upper_Band']
            df['超卖'] = df['close'] < df['Lower_Band']
            
            # 提取异常点数据
            overbought = df[df['超买'] == True].copy()
            oversold = df[df['超卖'] == True].copy()
            
            # 生成异常点统计表格
            result_text = f"""
**布林带异常检测结果 ({ts_code})**
- 检测时间范围: {start_date} 至 {end_date}
- 数据周期: 20日移动平均线 + 2σ标准差
- 总交易日数: {len(df)} 天

**超买点统计 (价格突破上轨):**
- 超买次数: {len(overbought)} 次
- 超买比例: {len(overbought)/len(df)*100:.2f}%

**超卖点统计 (价格跌破下轨):**
- 超卖次数: {len(oversold)} 次
- 超卖比例: {len(oversold)/len(df)*100:.2f}%
"""
            
            # 生成超买点表格
            if len(overbought) > 0:
                overbought_table = overbought[['trade_date', 'close', 'Upper_Band']].copy()
                overbought_table['trade_date'] = overbought_table['trade_date'].dt.strftime('%Y-%m-%d')
                overbought_table['close'] = overbought_table['close'].round(2)
                overbought_table['Upper_Band'] = overbought_table['Upper_Band'].round(2)
                overbought_table.columns = ['交易日期', '收盘价', '布林上轨']
                overbought_md = f"\n**超买点详情:**\n{overbought_table.to_markdown(index=False)}"
            else:
                overbought_md = "\n**超买点详情:** 无超买点"
            
            # 生成超卖点表格
            if len(oversold) > 0:
                oversold_table = oversold[['trade_date', 'close', 'Lower_Band']].copy()
                oversold_table['trade_date'] = oversold_table['trade_date'].dt.strftime('%Y-%m-%d')
                oversold_table['close'] = oversold_table['close'].round(2)
                oversold_table['Lower_Band'] = oversold_table['Lower_Band'].round(2)
                oversold_table.columns = ['交易日期', '收盘价', '布林下轨']
                oversold_md = f"\n**超卖点详情:**\n{oversold_table.to_markdown(index=False)}"
            else:
                oversold_md = "\n**超卖点详情:** 无超卖点"
            
            # 生成布林带可视化图表
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'boll_{ts_code}_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            
            plt.figure(figsize=(12, 8))
            
            # 绘制收盘价和布林带
            plt.plot(df['trade_date'], df['close'], label='收盘价', color='blue', linewidth=1.5)
            plt.plot(df['trade_date'], df['MA20'], label='20日移动平均线', color='orange', linewidth=1.5)
            plt.plot(df['trade_date'], df['Upper_Band'], label='布林上轨 (+2σ)', color='red', linestyle='--', alpha=0.7)
            plt.plot(df['trade_date'], df['Lower_Band'], label='布林下轨 (-2σ)', color='green', linestyle='--', alpha=0.7)
            
            # 标记异常点
            if len(overbought) > 0:
                plt.scatter(overbought['trade_date'], overbought['close'], 
                           color='red', s=50, marker='^', label='超买点', zorder=5)
            
            if len(oversold) > 0:
                plt.scatter(oversold['trade_date'], oversold['close'], 
                           color='green', s=50, marker='v', label='超卖点', zorder=5)
            
            plt.title(f'{ts_code} 布林带异常检测 (20日+2σ)')
            plt.xlabel('日期')
            plt.ylabel('价格')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 横坐标自动稀疏显示
            if len(df) > 20:
                step = max(1, len(df) // 10)
                show_idx = list(range(0, len(df), step))
                show_labels = [df['trade_date'].iloc[i].strftime('%Y-%m-%d') for i in show_idx]
                plt.xticks([df['trade_date'].iloc[i] for i in show_idx], show_labels, rotation=45)
            else:
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            img_path = os.path.join('image_show', filename)
            img_md = f'\n![布林带异常检测]({img_path})'
            
            return f"{result_text}{overbought_md}{oversold_md}{img_md}"
            
        except Exception as e:
            return f'布林带检测出错: {str(e)}'

@register_tool('prophet_analysis')
class ProphetAnalysisTool(BaseTool):
    description = '使用Prophet模型对股票进行周期性分析，分析trend、weekly、yearly趋势，并可视化展示。'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填',
            'required': True
        },
        {
            'name': 'start_date',
            'type': 'string',
            'description': '开始日期(YYYY-MM-DD格式)，可选，默认一年前',
            'required': False
        },
        {
            'name': 'end_date',
            'type': 'string',
            'description': '结束日期(YYYY-MM-DD格式)，可选，默认今天',
            'required': False
        }
    ]
    
    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        ts_code = args['ts_code']
        start_date = args.get('start_date')
        end_date = args.get('end_date')
        
        # 如果没有指定日期，默认使用过去一年
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 连接MySQL，获取历史收盘价
        engine = create_engine(
            f"mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/stock?charset=utf8mb4",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        
        sql = f"""
            SELECT trade_date, close FROM stock_price
            WHERE ts_code = '{ts_code}' AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date ASC
        """
        
        try:
            df = pd.read_sql(sql, engine)
            if len(df) < 60:
                return f'历史数据不足，至少需要60个交易日的数据进行Prophet周期性分析。当前只有{len(df)}条数据。'
            
            # 数据预处理
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 准备Prophet数据格式 (ds: 日期, y: 值)
            prophet_df = df[['trade_date', 'close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # 创建Prophet模型
            model = Prophet(
                yearly_seasonality=True,  # 年度季节性
                weekly_seasonality=True,  # 周度季节性
                daily_seasonality=False,  # 日度季节性（股票数据通常不需要）
                seasonality_mode='multiplicative',  # 乘法季节性
                changepoint_prior_scale=0.05,  # 趋势变化点先验
                seasonality_prior_scale=10.0   # 季节性先验
            )
            
            # 拟合模型
            model.fit(prophet_df)
            
            # 生成未来预测数据（用于分解）
            future = model.make_future_dataframe(periods=0)  # 不预测未来，只分析历史
            forecast = model.predict(future)
            
            # 获取分解结果
            decomposition = model.plot_components(forecast)
            
            # 保存分解图
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'prophet_decomp_{ts_code}_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            decomposition.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成趋势分析图
            trend_filename = f'prophet_trend_{ts_code}_{int(time.time()*1000)}.png'
            trend_save_path = os.path.join(save_dir, trend_filename)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 上图：原始数据和拟合结果
            ax1.plot(prophet_df['ds'], prophet_df['y'], label='实际收盘价', color='blue', alpha=0.7)
            ax1.plot(forecast['ds'], forecast['yhat'], label='Prophet拟合', color='red', linewidth=2)
            ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                           color='red', alpha=0.2, label='置信区间')
            ax1.set_title(f'{ts_code} Prophet模型拟合结果')
            ax1.set_ylabel('收盘价')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 下图：趋势分解
            ax2.plot(forecast['ds'], forecast['trend'], label='长期趋势', color='green', linewidth=2)
            ax2.plot(forecast['ds'], forecast['trend'] + forecast['yearly'], label='趋势+年度季节性', color='orange', alpha=0.7)
            ax2.plot(forecast['ds'], forecast['trend'] + forecast['yearly'] + forecast['weekly'], 
                    label='趋势+年度+周度季节性', color='purple', alpha=0.7)
            ax2.set_title('趋势分解分析')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('价格')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(trend_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 分析结果统计
            trend_slope = (forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]) / len(forecast)
            trend_direction = "上升" if trend_slope > 0 else "下降"
            
            # 计算季节性强度
            yearly_strength = forecast['yearly'].std() / forecast['yhat'].std() * 100
            weekly_strength = forecast['weekly'].std() / forecast['yhat'].std() * 100
            
            # 生成分析报告
            analysis_report = f"""
**Prophet周期性分析结果 ({ts_code})**
- 分析时间范围: {start_date} 至 {end_date}
- 数据点数量: {len(df)} 个交易日

**趋势分析:**
- 长期趋势方向: {trend_direction}
- 趋势斜率: {trend_slope:.4f} (每日变化)
- 趋势强度: {abs(trend_slope):.4f}

**季节性分析:**
- 年度季节性强度: {yearly_strength:.2f}%
- 周度季节性强度: {weekly_strength:.2f}%

**模型解释:**
- 长期趋势: 反映股票价格的长期发展方向
- 年度季节性: 反映一年内的周期性波动模式
- 周度季节性: 反映一周内的交易模式差异
- 残差: 无法被趋势和季节性解释的随机波动
"""
            
            # 生成趋势统计表格
            trend_stats = pd.DataFrame({
                '指标': ['起始趋势值', '结束趋势值', '趋势变化', '年度季节性强度', '周度季节性强度'],
                '数值': [
                    f"{forecast['trend'].iloc[0]:.2f}",
                    f"{forecast['trend'].iloc[-1]:.2f}",
                    f"{forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]:.2f}",
                    f"{yearly_strength:.2f}%",
                    f"{weekly_strength:.2f}%"
                ]
            })
            
            trend_table = trend_stats.to_markdown(index=False)
            
            # 返回结果
            decomp_img_path = os.path.join('image_show', filename)
            trend_img_path = os.path.join('image_show', trend_filename)
            
            decomp_img_md = f'![Prophet分解图]({decomp_img_path})'
            trend_img_md = f'![Prophet趋势分析]({trend_img_path})'
            
            return f"{analysis_report}\n\n{trend_table}\n\n{decomp_img_md}\n\n{trend_img_md}"
            
        except Exception as e:
            return f'Prophet分析出错: {str(e)}'

def init_agent_service():
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='股票查询助手',
            description='股票历史价格查询与分析',
            system_message=system_prompt,
            function_list=['exc_sql', 'arima_stock', 'boll_detection', 'prophet_analysis'],
            files = ['./faq.txt']
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    try:
        bot = init_agent_service()
        messages = []
        while True:
            try:
                query = input('user question: ')
                file = input('file url (press enter if no file): ').strip()
                if not query:
                    print('user question cannot be empty！')
                    continue
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})
                print("正在处理您的请求...")
                response = []
                for resp in bot.run(messages):
                    print('bot response:', resp)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    try:
        print("正在启动 Web 界面...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '查询2024年全年贵州茅台的收盘价走势',
                '统计2024年4月国泰君安的日均成交量',
                '对比2024年中芯国际和贵州茅台的涨跌幅',
                '预测贵州茅台未来7天的收盘价',
                '检测贵州茅台过去一年的布林带异常点',
                '使用Prophet分析贵州茅台的周期性趋势',
                '分析600519.SH的年度和周度季节性模式'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

if __name__ == '__main__':
    app_gui()  # 默认启动Web界面 
    #app_tui() 