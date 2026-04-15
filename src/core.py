def calculate_mean(data):
    """Calculate the mean of a list of numbers.

    Args:
        data (list): A list of lists of numbers.
    Returns:
        list: A list of means corresponding to each inner list.
    """
    data = data[[list(row) for row in zip(*data)]]
    mean_list = []
    for row in data:
        mean = sum(row) / len(row)
        mean_list.append(mean)

    return mean_list

def remove_mean(data):
    """Remove the mean from each element in a list of lists of numbers.

    Args:
        data (list): A list of lists of numbers.
    Returns:
        list: A list of lists with the mean removed from each element.
    """
    data = data[[list(row) for row in zip(*data)]]
    mean_list = calculate_mean(data)
    new_data = []
    for i in range(len(mean_list)):
        normalized_data = [data[i][j] - mean_list[i] for j in range(len(data[i]))]
        new_data.append(normalized_data)
    
    new_data = [list(row) for row in zip(*new_data)]

    return new_data

def covariance_matrix(data):
    """Calculate the covariance matrix of a list of lists of numbers.

    Args:
        data (list): A list of lists of numbers.
    Returns:
        list: A covariance matrix as a list of lists.
    """
    normalized_data = remove_mean(data)
    normalized_trans_data = normalized_data[[list(row) for row in zip(*normalized_data)]]
    feature_sum = [sum(row) for row in normalized_trans_data]
    for i in range(len(feature_sum)):
        for j in range
        feature_sum[i] = feature_sum[i] / (len(normalized_data) - 1)
