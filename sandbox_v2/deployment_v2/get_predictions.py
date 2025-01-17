import numpy as np
from read_models import read_models

abstract_certain_feat_xgb, abstract_certain_vect_xgb, abstract_certain_vect_nn, \
    abstract_subject_feat_xgb, abstract_subject_vect_xgb, abstract_subject_vect_nn, \
career_feat_xgb, career_vect_xgb, career_vect_nn, \
education_vect_nn, specific_feat_xgb, specific_vect_xgb, specific_vect_nn, \
time_bound_feat_xgb, time_bound_vect_xgb, time_bound_vect_nn, topic_art_vect_xgb, \
topic_art_vect_nn, topic_career_vect_xgb, topic_career_vect_nn, topic_community_vect_xgb, \
topic_community_vect_nn, topic_fixing_vect_xgb, topic_fixing_vect_nn, topic_habits_vect_xgb, \
topic_habits_vect_nn, topic_hard_skill_vect_xgb, topic_hard_skill_vect_nn, topic_health_vect_xgb, \
topic_health_vect_nn, topic_knowledge_vect_xgb, topic_knowledge_vect_nn, topic_soft_skill_vect_xgb, \
topic_soft_skill_vect_nn, topic_subjectivity_vect_xgb, topic_subjectivity_vect_nn, topic_tool_vect_xgb, \
topic_tool_vect_nn, topics_tesaurus = read_models()

topic_map = {
    'art_pred':'Искусство',
    'knowledge_pred':'Знания',
    'hard_skill_pred':'Hard skills',
    'soft_skill_pred':'Soft skills',
    'subjectivity_pred':'Субъективизация',
    'tool_pred':'Инструменты/Прикладное',
    'health_pred':'Здоровье',
    'habits_pred':'Привычки',
    'fixing_pred':'Фиксация',
    'community_pred':'Общество/Семья',
    'career_pred':'Карьера',
}

def algo_vote(predictions:list):
    zero_counter = int()
    one_counter = int()
    for i in predictions:
        if i == 0:
            zero_counter += 1
        else:
            one_counter += 1
    if zero_counter > one_counter:
        return 0
    else:
        return 1
    # return predictions[1]

def get_prediction(model1, model2, model3, features, vectors, net, raw_domain):
    pred1 = model1.predict(features)
    pred2 = model2.predict(vectors)
    pred3 = np.argmax(model3.predict(net), axis = -1)
    final_pred = algo_vote([pred1, pred2, pred3])
    return final_pred

def predict_topic(model1, model2, vectors, net, key, raw_domain, tesaurus=topics_tesaurus):
    pred1 = model1.predict(vectors)[0]
    pred2 = np.argmax(model2.predict(net), axis = -1)[0]
    key_words = topics_tesaurus[key]
    pred3 = 0
    for word in key_words:
        if word in raw_domain:
            pred3 = 1
            break
    final_pred = algo_vote([pred1, pred2, pred3])
    return final_pred
    # return pred1
    
def predict(features, vectors, net, raw_domain, input_df):
    # SMART
        # is specific
    specific_pred = get_prediction(specific_feat_xgb, specific_vect_xgb, specific_vect_nn, features, vectors, net, raw_domain)
    if specific_pred == 1:
        specific_pred = 'Цель сформулирована конкретно/specific'
    else:
        specific_pred = 'Цель сформулирована НЕ конкретно/specific'

        # is attainable
    obstackles = input_df.loc[:, 'are_obstackles_expected'].tolist()[0]
    if obstackles == 0:
        attainable_pred = 'Цель достижима/attainable'
    else:
        attainable_pred = 'Цель НЕ достижима/attainable'

        # is time bound
    time_bound_pred = get_prediction(time_bound_feat_xgb, time_bound_vect_xgb, time_bound_vect_nn, features, vectors, net, raw_domain)
    if time_bound_pred == 1:
        time_bound_pred = 'Цель ограничена во времени/time-bound'
    else:
        time_bound_pred = 'Цель НЕ ограничена во времени/time-bound'

    SMART_pred = [specific_pred, attainable_pred, time_bound_pred]

    # is education
    education_pred = np.argmax(education_vect_nn.predict(net), axis = -1)[0]
    if education_pred == 1:
        education_pred = 'Цель имеет отношение к образованию'
    else:
        education_pred = 'Цель НЕ имеет отношения к образованию'

    # is career
    career_pred = get_prediction(career_feat_xgb, career_vect_xgb, career_vect_nn, features, vectors, net, raw_domain)
    if career_pred == 1:
        career_pred = 'Цель имеет отношения к карьере'
    else:
        career_pred = 'Цель НЕ имеет отношения к карьере'

    edu_car_pred = [education_pred, career_pred]

    # is certain
    abstract_pred = ''
    certain_pred = get_prediction(abstract_certain_feat_xgb, abstract_certain_vect_xgb, 
                                    abstract_certain_vect_nn, features, vectors, net, raw_domain)
    if certain_pred == 1:
        abstract_pred = 'Цель сформулирована конкретно'
    else:
        # is subject|abstract
        subject_pred = get_prediction(abstract_subject_feat_xgb, abstract_subject_vect_xgb, abstract_subject_vect_nn, features, vectors, net, raw_domain)
        if subject_pred == 1:
            abstract_pred = 'Цель сформулирована предметно'
        else:
            abstract_pred = 'Цель сформулирована абстрактно'

    # TOPICS
    art_pred = predict_topic(topic_art_vect_xgb, topic_art_vect_nn, vectors, net, 'label_attractor_art', 
                                raw_domain, tesaurus=topics_tesaurus)

    knowledge_pred = predict_topic(topic_knowledge_vect_xgb, topic_knowledge_vect_nn, vectors, net, 'label_attractor_knowledge', 
                                raw_domain, tesaurus=topics_tesaurus)

    hard_skill_pred = predict_topic(topic_hard_skill_vect_xgb, topic_hard_skill_vect_nn, vectors, net, 'label_attractor_hard_skill', 
                                raw_domain, tesaurus=topics_tesaurus)

    soft_skill_pred = predict_topic(topic_soft_skill_vect_xgb, topic_soft_skill_vect_nn, vectors, net, 'label_attractor_soft_skill', 
                                raw_domain, tesaurus=topics_tesaurus)

    subjectivity_pred = predict_topic(topic_subjectivity_vect_xgb, topic_subjectivity_vect_nn, vectors, net, 'label_attractor_subjectivity', 
                                raw_domain, tesaurus=topics_tesaurus)

    tool_pred = predict_topic(topic_tool_vect_xgb, topic_tool_vect_nn, vectors, net, 'label_attractor_tool', 
                                raw_domain, tesaurus=topics_tesaurus)

    health_pred = predict_topic(topic_health_vect_xgb, topic_health_vect_nn, vectors, net, 'label_attractor_health', 
                                raw_domain, tesaurus=topics_tesaurus)

    habits_pred = predict_topic(topic_habits_vect_xgb, topic_habits_vect_nn, vectors, net, 'label_attractor_habits', 
                                raw_domain, tesaurus=topics_tesaurus)

    fixing_pred = predict_topic(topic_fixing_vect_xgb, topic_fixing_vect_nn, vectors, net, 'label_attractor_fixing', 
                                raw_domain, tesaurus=topics_tesaurus)

    community_pred = predict_topic(topic_community_vect_xgb, topic_community_vect_nn, vectors, net, 'label_attractor_community', 
                                raw_domain, tesaurus=topics_tesaurus)

    career_pred = predict_topic(topic_career_vect_xgb, topic_career_vect_nn, vectors, net, 'label_attractor_career', 
                                raw_domain, tesaurus=topics_tesaurus)

    topics_pred = []
    for value, key in [(art_pred, 'art_pred'), (knowledge_pred, 'knowledge_pred'),
                (hard_skill_pred, 'hard_skill_pred'), (soft_skill_pred, 'soft_skill_pred'),
                (subjectivity_pred, 'subjectivity_pred'), (tool_pred, 'tool_pred'),
                (health_pred, 'health_pred'), (habits_pred, 'habits_pred'), 
                (fixing_pred, 'fixing_pred'), (community_pred, 'community_pred'), 
                (career_pred, 'career_pred')]:
        if value == 1:
            topics_pred.append(topic_map[key])

    return SMART_pred, edu_car_pred, abstract_pred, topics_pred