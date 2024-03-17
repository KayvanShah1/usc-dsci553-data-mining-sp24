from pyspark import SparkContext
import time
import sys


def get_predictions(test_review, user_to_business):

    user = test_review[0][0]

    neighbors = test_review[1]
    # print('neighbors type:', type(neighbors))
    # print('len neighbors:', len(neighbors))

    if len(neighbors) == 2:
        # return neighbors[1]
        #        print('SINGLE NEIGHBORS:', neighbors)
        return neighbors[1]

    test_user_ratings = user_to_business[user]

    pred = 0
    sum_weights = 0

    for n in neighbors:
        weight = n[1]
        business = n[0]
        # print('N:', n)
        given_rating = test_user_ratings[business]
        pred += weight * float(given_rating)
        sum_weights += abs(weight)

    if len(neighbors) == 1:
        #        print('SINGLE NEIGHBOR', neighbors)
        sum_weights = 1
        pred = 1
    pred = pred / sum_weights

    if pred < 1:
        pred = 1.0

    if pred > 5:
        pred = 5.0

    return pred


def join_values(tuple_val):

    tuple_of_dictionaries = tuple_val[1]

    init = tuple_of_dictionaries[0]

    for i in tuple_of_dictionaries[1:]:
        init = {**init, **i}

    return (tuple_val[0], init)


def preprocessing(rdd):
    first_element = rdd.first()
    rdd = rdd.filter(lambda line: line != first_element)
    return rdd


def calculate_similarity(ratings, other_ratings):

    average_rating = sum(ratings) / len(ratings)
    other_average_rating = sum(other_ratings) / len(other_ratings)

    pearson_value = 0
    numerator = 0
    d1 = 0
    d2 = 0

    for i in range(len(ratings)):
        numerator += (ratings[i] - average_rating) * (other_ratings[i] - other_average_rating)
        d1 += (ratings[i] - average_rating) ** 2
        d2 += (other_ratings[i] - other_average_rating) ** 2

    d1 = d1**0.5
    d2 = d2**0.5

    try:

        pearson_value = numerator / (d1 * d2)
        return float(pearson_value**2.5)

    except:
        return 0.9


def pearson_similarity(test_review, business_to_user, user_to_business, neighbors, threshold):

    default_weight = 0.9
    test_business = test_review[1]
    test_user = test_review[0]

    if test_user in user_to_business:
        test_user_rated_businesses = user_to_business[test_user]
        test_user_rated_businesses = {k for k, v in test_user_rated_businesses.items()}

    else:
        user_ratings = business_to_user[test_business]
        user_ratings = [float(v) for k, v in user_ratings.items()]
        return (tuple(test_review), sum(user_ratings) / len(user_ratings))
    #        return (tuple(test_review), 3.0)

    if test_business in business_to_user:

        test_business_reviewers = business_to_user[test_business]
        test_business_reviewers = {k for k, v in test_business_reviewers.items()}

    else:
        user_ratings = user_to_business[test_user]
        user_ratings = [float(v) for k, v in user_ratings.items()]
        return (tuple(test_review), sum(user_ratings) / len(user_ratings))
    #        return (tuple(test_review), default_weight)
    #        return (tuple(test_review), 5.0)

    similarities = []

    if len(test_user_rated_businesses) < neighbors:
        user_ratings = user_to_business[test_user]
        user_ratings = [float(v) for k, v in user_ratings.items()]
        return (tuple(test_review), sum(user_ratings) / len(user_ratings))

    for i in test_user_rated_businesses:
        other_business_rating = []
        test_business_rating = []
        if i == test_business:
            continue

        other_users = business_to_user[i]

        other_users = {k for k, v in other_users.items()}

        common_users = other_users.intersection(test_business_reviewers)

        for j in common_users:

            try:
                other_business_rating.append(float(user_to_business[j][i]))
                test_business_rating.append(float(user_to_business[j][test_business]))

            except KeyError:

                print("KeyError")

        if len(test_business_rating) > 1:
            similarity = calculate_similarity(test_business_rating, other_business_rating)
            similarities.append((i, similarity))

        else:
            similarities.append((i, default_weight))

    if len(similarities) < threshold:
        user_ratings = user_to_business[test_user]
        user_ratings = [float(v) for k, v in user_ratings.items()]
        return (tuple(test_review), sum(user_ratings) / len(user_ratings))

    if len(similarities) > neighbors:
        similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True)
        similarities = similarities[:neighbors]
    return tuple(similarities)


def main():

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    neighbors = 50
    threshold = 149

    sc = SparkContext()
    train_rdd = sc.textFile(train_path).map(lambda x: x.split(","))
    test_rdd = sc.textFile(test_path).map(lambda x: x.split(",")).cache()

    business_to_user = train_rdd.map(lambda review: (review[1], {review[0]: review[2]}))
    user_to_business = train_rdd.map(lambda review: (review[0], {review[1]: review[2]}))

    train_rdd = preprocessing(train_rdd)
    test_rdd = preprocessing(test_rdd)

    business_to_user = business_to_user.groupByKey().mapValues(tuple)
    business_to_user = business_to_user.map(lambda x: join_values(x)).collectAsMap()

    user_to_business = user_to_business.groupByKey().mapValues(tuple)
    user_to_business = user_to_business.map(lambda x: join_values(x)).collectAsMap()

    similarity_rdd = test_rdd.map(
        lambda review: (review, pearson_similarity(review, business_to_user, user_to_business, neighbors, threshold))
    )

    prediction_rdd = (
        similarity_rdd.map(lambda sim: (sim[0], get_predictions(sim, user_to_business)))
        .map(lambda line: (line[0][0], line[0][1], line[1]))
        .collect()
    )

    with open(output_path, "w") as f:
        string = "user_id, business_id, prediction \n"

        f.write(string)

        # string = ''

        for line in prediction_rdd:
            string = line[0] + "," + line[1] + "," + str(line[2]) + "\n"
            f.write(string)


start = time.time()
main()
end = time.time()

print("Duration:", end - start)
