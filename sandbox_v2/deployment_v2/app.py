import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField

from input_adj_pipeline import create_featured_datasets
from get_predictions import *


class UserForm(FlaskForm):
    goal_name = TextField('Формулировка цели*')
    goal_result = TextField('Есть ли у Вас образ результата по этой цели?')
    goal_type = TextField('К какому типу запроса относится Ваша цель?')
    goal_first_step = TextField('Каким может быть первый шаг для достижения данной цели? С чего бы Вы начали?')
    goal_domain = TextField('К какой тематической области относится Ваша цель?')
    goal_obstacle = TextField('Какие Вы видите преграды для достижения этой цели? Что может помешать?')
    goal_time = TextField('Сколько времени, как Вам кажется, может занять достижение Вами данной цели?')
    submit = SubmitField('Получить оценку')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'pers_target_secretkey'


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UserForm()
    if form.validate_on_submit():
        session['goal_name'] = form.goal_name.data
        session['goal_result'] = form.goal_result.data
        session['goal_type'] = form.goal_type.data
        session['goal_first_step'] = form.goal_first_step.data
        session['goal_domain'] = form.goal_domain.data
        session['goal_obstacle'] = form.goal_obstacle.data
        session['goal_time'] = form.goal_time.data
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    if not session['goal_name']:
        return redirect(url_for('index'))
    else:
        topic_raw = session['goal_domain']
        input_ = {
        'goal_name':session['goal_name'],
        'goal_result':session['goal_result'],
        'goal_type':session['goal_type'],
        'goal_first_step':session['goal_first_step'],
        'goal_domain':session['goal_domain'],
        'goal_obstacle':session['goal_obstacle'],
        'goal_time':session['goal_time'],
    }
        features, vectors, net, input_df = create_featured_datasets(input_)
        SMART_pred, edu_car_pred, abstract_pred, topics_pred = predict(features, vectors, net, topic_raw, input_df)

        return render_template('prediction.html', smart=SMART_pred, edu_car=edu_car_pred, 
                                abstract=abstract_pred, topics=topics_pred)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
