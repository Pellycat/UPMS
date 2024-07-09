from django.shortcuts import render


from django.shortcuts import render, redirect
from django.http import JsonResponse
import pandas as pd
import numpy as np
from .models import Lasso_Predicted_Return,EN_Predicted_Return,Rd_Predicted_Return,Real_Return


def data_process(data, capital, num, start, end):
    # 确保 Date 列为 datetime 类型
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.drop_duplicates(['Date', 'Stock'])
    # 设置 Date 列为索引
    data.set_index('Date', inplace=True)

    # 使用 pivot 函数重新排列数据
    pivot_df = data.pivot(columns='Stock', values='Predicted_Return')

    # 重置索引以将日期恢复为列
    pivot_df.reset_index(inplace=True)

    df = pivot_df

    df.index = pd.to_datetime(df.index)

    # 设置 'Date' 列为索引，并按降序排序
    df = df.set_index('Date').sort_index(ascending=False)

    # 由于rolling方法计算的是向上的窗口，所以日期索引必须要按降序排列
    rolling_returns = df.rolling(window=30).sum()
    number = int(num)
    top_stocks_df = pd.DataFrame(index=rolling_returns.index, columns=[f'Top_{i + 1}' for i in range(number)])

    for date in rolling_returns.index:
        if not rolling_returns.loc[date].isna().all():
            top_stocks = rolling_returns.loc[date].nlargest(number).index
            top_stocks_df.loc[date] = top_stocks

    # 取消设置日期列为索引，保留日期列作为普通列
    top_stocks_df = top_stocks_df.reset_index(drop=False)
    top_stocks_df = top_stocks_df.set_index('Date').sort_index(ascending=True)
    top_stocks_df.index = pd.to_datetime(top_stocks_df.index)

    # 创建 yearmonth 列
    top_stocks_df['yearmonth'] = top_stocks_df.index.strftime('%Y-%m')

    # 保留每月第一天的数据
    top_stocks_df = top_stocks_df.drop_duplicates(subset='yearmonth', keep='first')

    print(top_stocks_df)
    top_stocks_df.dropna(how='all', inplace=True)

    # 取消设置日期列为索引，保留日期列作为普通列
    top_stocks_df = top_stocks_df.reset_index(drop=False)
    print("top_stocks_df")
    print(top_stocks_df)

    # 得到真实超额收益率，日度
    result = Real_Return.objects.all()
    market = pd.DataFrame(list(result.values()))
    market['Ex_Return'] = market['Ex_Return'].astype(float)
    market.set_index('Date', inplace=True)

    # 这里添加将索引转换为DatetimeIndex的代码
    market.index = pd.to_datetime(market.index)

    # 按股票代码进行分组并计算月度超额收益率
    monthly_excess_returns = market.groupby(['Stock', market.index.to_period('M')])['Ex_Return'].sum().unstack(level=0)

    # 确保列名是字符串
    monthly_excess_returns.columns = monthly_excess_returns.columns.astype(str)
    monthly_excess_returns.reset_index(inplace=True)

    # 转换为 Timestamp 格式
    monthly_excess_returns['Date'] = monthly_excess_returns['Date'].apply(lambda x: x.to_timestamp())

    print("monthly_excess_returns")
    print(monthly_excess_returns)
    dates = top_stocks_df['yearmonth'].unique()

    # 新建一个表格用来填充数据
    result_df = pd.DataFrame(index=dates, columns=[f'Top_{i + 1}' for i in range(number)])
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'Date'}, inplace=True)



    # 填充新表格
    for date in dates:
        if date in result_df['Date'].values:
            for i in range(number):
                stock = top_stocks_df.loc[top_stocks_df['yearmonth'] == date, f'Top_{i + 1}'].values[0]
                if pd.isna(stock):
                    result_df.loc[result_df['Date'] == date, f'Top_{i + 1}'] = np.nan
                else:
                    stock = str(stock)  # 确保 stock 是字符串
                    if stock in monthly_excess_returns.columns:
                        result_df.loc[result_df['Date'] == date, f'Top_{i + 1}'] = \
                        monthly_excess_returns.loc[monthly_excess_returns['Date'] == date, stock].values[0]
                    else:
                        result_df.loc[result_df['Date'] == date, f'Top_{i + 1}'] = np.nan
        else:
            print(f"日期 {date} 未在result_df中找到")
    print("result_df", result_df)
#-------------------------------------------------------------------------以上都no problem
    result_df.dropna(inplace=True)

    top_stocks_df.drop("yearmonth",axis=1,inplace=True)
    # 定义效用最大化函数
    def util_max(cov_matrix, expect_returns, gamma):
        Q = gamma * cov_matrix
        uhat = np.mat(expect_returns).T  # 转换为列向量
        c = -uhat
        I = np.mat(np.eye(len(uhat)))
        A = np.mat(np.ones((len(uhat), 1))).T
        b = np.mat(np.array([1])[None])

        z1 = A * np.linalg.inv(Q) * A.T
        z2 = np.linalg.inv(Q) * A.T
        z3 = z2 * np.linalg.inv(z1) * b
        z4 = A.T * np.linalg.inv(z1) * A * np.linalg.inv(Q)
        z5 = I - z4
        z6 = np.linalg.inv(Q) * z5 * c
        optimal_weights = z3 - z6
        return np.array(optimal_weights).flatten()  # 转换为一维数组

    # 定义最小方差最优化函数
    def min_var_opt(cov_matrix, expect_returns):
        uhat = expect_returns  # 每列均值，表示每个变量的超额收益率的平均值。
        A = np.mat(np.concatenate([uhat.reshape(-1, 1), np.ones(len(uhat)).reshape(-1, 1)], axis=1)).T
        up = np.mean(uhat)  # uhat的均值，表示所有变量超额收益率的平均值。
        b = np.mat(np.array([up, 1])[:, None])
        optimal_weights = np.linalg.inv(cov_matrix) * A.T * np.linalg.inv(A * np.linalg.inv(cov_matrix) * A.T) * b
        return np.array(optimal_weights).flatten()  # 转换为一维数组

    # 定义等权重分配函数
    def equal_weight_opt(num_assets):
        return np.ones(num_assets) / num_assets

    # 数据加载和预处理
    monthly_returns = monthly_excess_returns

    monthly_returns = monthly_returns.set_index('Date')
    # 将索引的date列转换为%Y-%m格式
    monthly_returns.index = pd.to_datetime(monthly_returns.index).strftime('%Y-%m')

    select = top_stocks_df
    #select.columns = ['date', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
    # 将date列转换为%Y-%m格式
    #select['date'] = pd.to_datetime(select['date']).dt.strftime('%Y-%m')

    # 读取年份月份

    year_months = pd.date_range(start=start, end=end, freq='ME').strftime('%Y%m')

    # 初始化权重矩阵
    weights_min_var = {}  # 最小化方差
    weights_util_max = {}  # 最大化效用
    weights_equal = {}  # 等权重分配
    # 设置风险厌恶系数
    gamma = 3
    tau = 1

    # 初始化用于存储样本外统计量的字典
    excess_returns_stats_min_var = {}
    excess_returns_stats_util_max = {}
    excess_returns_stats_equal = {}

    # 初始化用于存储当月投资组合超额收益率的变量
    monthly_excess_returns_min_var = []
    monthly_excess_returns_util_max = []
    monthly_excess_returns_equal = []
    monthly_dates = []

    # 初始化covariance_matrices_by_month示例
    covariance_matrices_by_month = {ym: {'cov_shrinkage': np.eye(10)} for ym in year_months}  # 替换为实际的协方差矩阵数据

    # 将月度超额收益率数据转换为 DataFrame
    monthly_returns_df = monthly_excess_returns.copy()
    monthly_returns_df['Date'] = pd.to_datetime(monthly_returns_df['Date'])
    monthly_returns_df.set_index('Date', inplace=True)

    # 计算超额收益率的历史均值
    excess_returns_mean = monthly_returns_df.mean(axis=0)

    # 遍历需要计算的日期
    t = 0
    for yearmonth in year_months:
        date = str(yearmonth)[:4] + '-' + str(yearmonth)[4:6]
        print(f"Processing date: {date}")

        # 获取当前日期的选择股票代码
        selected_stocks = select[select['Date'] == date].iloc[:, 1:].values.flatten()
        selected_stocks = [str(stock) for stock in selected_stocks]
        selected_stocks = [stock for stock in selected_stocks if stock in monthly_returns.columns]  # 过滤掉不在数据中的股票
        print("Selected stocks: ", selected_stocks)

        current_data = monthly_returns[selected_stocks].loc[:date]
        current_data.fillna(0,inplace=True)
        print("current_data")
        print(current_data)
        # 跳过没有数据的月份
        if current_data.empty:
            print(f"No data available for date: {date}")
            continue

        # 剔除高度相关的资产
        # current_data = remove_highly_correlated_assets(current_data)

        # 跳过剔除后没有足够资产的月份
        if current_data.shape[1] < 2:
            print(f"Not enough assets after removing highly correlated ones for date: {date}")
            continue

        # 将当前日期的超额收益率转换为矩阵R
        R = current_data.values

        # 计算协方差矩阵和预期收益
        cov_t = np.cov(R, rowvar=False)
        expect_returns = np.mean(R, axis=0)

        try:
            # 计算最小方差组合权重
            weights_min_var[date] = min_var_opt(cov_t, expect_returns)

            # 计算最大效用组合权重
            weights_util_max[date] = util_max(cov_t, expect_returns, gamma)

            # 计算等权重组合权重
            weights_equal[date] = equal_weight_opt(len(selected_stocks))
        except np.linalg.LinAlgError:
            # 跳过奇异矩阵月份
            print(f"Singular matrix encountered for date: {date}")
            continue

        # 计算样本外统计量
        next_month_data = monthly_returns[selected_stocks].loc[date].values
        if next_month_data.size > 0:
            if date in weights_min_var and date in weights_util_max and date in weights_equal:
                if weights_min_var[date].shape[0] == next_month_data.shape[0]:
                    excess_returns_min_var_sample_out = np.dot(weights_min_var[date], next_month_data)
                    excess_returns_util_max_sample_out = np.dot(weights_util_max[date], next_month_data)
                    excess_returns_equal_sample_out = np.dot(weights_equal[date], next_month_data)

                    # 存储样本外统计量
                    excess_returns_stats_min_var[date] = np.mean(excess_returns_min_var_sample_out)
                    excess_returns_stats_util_max[date] = np.mean(excess_returns_util_max_sample_out)
                    excess_returns_stats_equal[date] = np.mean(excess_returns_equal_sample_out)

        # 计算当月投资组合超额收益率
        excess_returns_t = np.array(next_month_data).reshape(len(next_month_data), 1)
        Q = gamma * cov_t
        A = np.ones((1, len(next_month_data)))
        b = 1
        c = -expect_returns.reshape((len(expect_returns), 1))

        temp1 = np.linalg.inv(A @ np.linalg.inv(Q) @ A.T)
        temp2 = np.eye(len(expect_returns)) - A.T @ temp1 @ A @ np.linalg.inv(Q)
        w1 = np.linalg.inv(Q) @ A.T @ temp1
        w2 = np.linalg.inv(Q) @ temp2 @ c
        w_star = w1 - w2

        optimal_weights = w_star.flatten()
        monthly_excess_returns = np.dot(w_star.T, excess_returns_t).flatten()

        monthly_excess_returns_min_var.append(np.dot(weights_min_var[date], next_month_data))
        monthly_excess_returns_util_max.append(np.dot(weights_util_max[date], next_month_data))
        monthly_excess_returns_equal.append(np.dot(weights_equal[date], next_month_data))
        monthly_dates.append(date)

        t += 1

    # 确保所有列表的长度一致
    min_length = min(len(monthly_dates), len(monthly_excess_returns_min_var), len(monthly_excess_returns_util_max),
                     len(monthly_excess_returns_equal))
    monthly_dates = monthly_dates[:min_length]
    monthly_excess_returns_min_var = monthly_excess_returns_min_var[:min_length]
    monthly_excess_returns_util_max = monthly_excess_returns_util_max[:min_length]
    monthly_excess_returns_equal = monthly_excess_returns_equal[:min_length]

    # 输出结果
    print(weights_min_var)
    print(weights_util_max)
    print(weights_equal)
    print(excess_returns_stats_min_var)
    print(excess_returns_stats_util_max)
    print(excess_returns_stats_equal)

    # 将当月投资组合超额收益率转换为DataFrame并输出
    monthly_excess_returns_df = pd.DataFrame({
        'Date': monthly_dates,
        'Min_Var_Excess_Returns': monthly_excess_returns_min_var,
        'Util_Max_Excess_Returns': monthly_excess_returns_util_max,
        'Equal_Excess_Returns': monthly_excess_returns_equal
    })
    monthly_excess_returns_df.to_csv('monthly_excess_returns_portfolio.csv', index=False)

    print("Monthly excess returns saved to monthly_excess_returns_portfolio.csv")

    # 找到最长的权重数组长度
    max_length = max(len(weights) for weights in weights_min_var.values())

    # 将权重数据转换为DataFrame并写入Excel文件
    weights_min_var_df = pd.DataFrame(
        {date: np.pad(weights, (0, max_length - len(weights)), 'constant', constant_values=np.nan)
         for date, weights in weights_min_var.items()}).T
    weights_util_max_df = pd.DataFrame(
        {date: np.pad(weights, (0, max_length - len(weights)), 'constant', constant_values=np.nan)
         for date, weights in weights_util_max.items()}).T
    weights_equal_df = pd.DataFrame(
        {date: np.pad(weights, (0, max_length - len(weights)), 'constant', constant_values=np.nan)
         for date, weights in weights_equal.items()}).T

    weights_min_var_html = weights_min_var_df.to_html(classes='table table-striped')
    weights_util_max_html = weights_util_max_df.to_html(classes='table table-striped')
    weights_equal_html = weights_equal_df.to_html(classes='table table-striped')

    return weights_min_var_html, weights_util_max_html, weights_equal_html


def choice_gupiao(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        model = request.POST.get('algorithmId')
        capital = request.POST.get('capital')
        num = request.POST.get('pool_num')
        start = request.POST.get('startTime')
        end = request.POST.get('endTime')
        print(name, model, capital, num, start, end)

        if model == "Lasso":
            result = Lasso_Predicted_Return.objects.filter(Date__range=[start, end])
            result_df = pd.DataFrame(list(result.values()))
            weights_min_var_html, weights_util_max_html, weights_equal_html = data_process(result_df, capital, num, start, end)
            return JsonResponse({'message': '数据已收到', 'start_date': start, 'end_date': end})

        elif model == "EN":
            result = EN_Predicted_Return.objects.filter(Date__range=[start, end])
            result_df = pd.DataFrame(list(result.values()))
            weights_min_var_html, weights_util_max_html, weights_equal_html = data_process(result_df, capital, num, start, end)
            return JsonResponse({'message': '数据已收到', 'start_date': start, 'end_date': end})
        elif model == "Ridge":
            result = Rd_Predicted_Return.objects.filter(Date__range=[start, end])
            result_df = pd.DataFrame(list(result.values()))
            weights_min_var_html, weights_util_max_html, weights_equal_html = data_process(result_df, capital, num, start, end)
            return JsonResponse({'message': '数据已收到', 'start_date': start, 'end_date': end})
        else:
            return JsonResponse({'message': '请选择合适的模型'})


    return render(request, 'inputadd.html')
