import os
import pickle
from tensorflow.keras.models import load_model

PATH = 'models/'

def read_models():

    with open(os.path.join(PATH, 'abstract_certain_feat_xgb.pkl'), 'rb') as f:
        abstract_certain_feat_xgb = pickle.load(f)

    with open(os.path.join(PATH, 'abstract_certain_vect_xgb.pkl'), 'rb') as f:
        abstract_certain_vect_xgb = pickle.load(f)

    abstract_certain_vect_nn = load_model(os.path.join(PATH, 'abstract_certain_vect_nn.h5'))

    with open(os.path.join(PATH, 'abstract_subject_feat_xgb.pkl'), 'rb') as f:
        abstract_subject_feat_xgb = pickle.load(f)

    with open(os.path.join(PATH, 'abstract_subject_vect_xgb.pkl'), 'rb') as f:
        abstract_subject_vect_xgb = pickle.load(f)

    abstract_subject_vect_nn = load_model(os.path.join(PATH, 'abstract_subject_vect_nn.h5'))

    with open(os.path.join(PATH, 'career_feat_xgb.pkl'), 'rb') as f:
        career_feat_xgb = pickle.load(f)

    with open(os.path.join(PATH, 'career_vect_xgb.pkl'), 'rb') as f:
        career_vect_xgb = pickle.load(f)

    career_vect_nn = load_model(os.path.join(PATH, 'career_vect_nn.h5'))

    education_vect_nn = load_model(os.path.join(PATH, 'education_vect_nn.h5'))

    with open(os.path.join(PATH, 'specific_feat_xgb.pkl'), 'rb') as f:
        specific_feat_xgb = pickle.load(f)

    with open(os.path.join(PATH, 'specific_vect_xgb.pkl'), 'rb') as f:
        specific_vect_xgb = pickle.load(f)

    specific_vect_nn = load_model(os.path.join(PATH, 'specific_vect_nn.h5'))

    with open(os.path.join(PATH, 'time_bound_feat_xgb.pkl'), 'rb') as f:
        time_bound_feat_xgb = pickle.load(f)

    with open(os.path.join(PATH, 'time_bound_vect_xgb.pkl'), 'rb') as f:
        time_bound_vect_xgb = pickle.load(f)

    time_bound_vect_nn = load_model(os.path.join(PATH, 'time_bound_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_art_vect_xgb.pkl'), 'rb') as f:
        topic_art_vect_xgb = pickle.load(f)

    topic_art_vect_nn = load_model(os.path.join(PATH, 'topic_art_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_career_vect_xgb.pkl'), 'rb') as f:
        topic_career_vect_xgb = pickle.load(f)

    topic_career_vect_nn = load_model(os.path.join(PATH, 'topic_career_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_community_vect_xgb.pkl'), 'rb') as f:
        topic_community_vect_xgb = pickle.load(f)

    topic_community_vect_nn = load_model(os.path.join(PATH, 'topic_community_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_fixing_vect_xgb.pkl'), 'rb') as f:
        topic_fixing_vect_xgb = pickle.load(f)

    topic_fixing_vect_nn = load_model(os.path.join(PATH, 'topic_fixing_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_habits_vect_xgb.pkl'), 'rb') as f:
        topic_habits_vect_xgb = pickle.load(f)

    topic_habits_vect_nn = load_model(os.path.join(PATH, 'topic_habits_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_hard_skill_vect_xgb.pkl'), 'rb') as f:
        topic_hard_skill_vect_xgb = pickle.load(f)

    topic_hard_skill_vect_nn = load_model(os.path.join(PATH, 'topic_hard_skill_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_health_vect_xgb.pkl'), 'rb') as f:
        topic_health_vect_xgb = pickle.load(f)

    topic_health_vect_nn = load_model(os.path.join(PATH, 'topic_health_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_knowledge_vect_xgb.pkl'), 'rb') as f:
        topic_knowledge_vect_xgb = pickle.load(f)

    topic_knowledge_vect_nn = load_model(os.path.join(PATH, 'topic_knowledge_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_soft_skill_vect_xgb.pkl'), 'rb') as f:
        topic_soft_skill_vect_xgb = pickle.load(f)

    topic_soft_skill_vect_nn = load_model(os.path.join(PATH, 'topic_soft_skill_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_subjectivity_vect_xgb.pkl'), 'rb') as f:
        topic_subjectivity_vect_xgb = pickle.load(f)

    topic_subjectivity_vect_nn = load_model(os.path.join(PATH, 'topic_subjectivity_vect_nn.h5'))

    with open(os.path.join(PATH, 'topic_tool_vect_xgb.pkl'), 'rb') as f:
        topic_tool_vect_xgb = pickle.load(f)

    topic_tool_vect_nn = load_model(os.path.join(PATH, 'topic_tool_vect_nn.h5'))

    with open(os.path.join(PATH, 'topics_tesaurus.pickle'), 'rb') as f:
        topics_tesaurus = pickle.load(f)

    return abstract_certain_feat_xgb, abstract_certain_vect_xgb, abstract_certain_vect_nn, \
        abstract_subject_feat_xgb, abstract_subject_vect_xgb, abstract_subject_vect_nn, \
        career_feat_xgb, career_vect_xgb, career_vect_nn, \
            education_vect_nn, specific_feat_xgb, specific_vect_xgb, specific_vect_nn, \
            time_bound_feat_xgb, time_bound_vect_xgb, time_bound_vect_nn, topic_art_vect_xgb, \
            topic_art_vect_nn, topic_career_vect_xgb, topic_career_vect_nn, topic_community_vect_xgb, \
            topic_community_vect_nn, topic_fixing_vect_xgb, topic_fixing_vect_nn, topic_habits_vect_xgb, \
            topic_habits_vect_nn, topic_hard_skill_vect_xgb, topic_hard_skill_vect_nn, topic_health_vect_xgb, \
            topic_health_vect_nn, topic_knowledge_vect_xgb, topic_knowledge_vect_nn, topic_soft_skill_vect_xgb, \
            topic_soft_skill_vect_nn, topic_subjectivity_vect_xgb, topic_subjectivity_vect_nn, topic_tool_vect_xgb, \
            topic_tool_vect_nn, topics_tesaurus

