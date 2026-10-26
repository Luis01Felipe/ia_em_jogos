from arena.pong.game import PongConfig
from arena.pong.imitation import evaluate, record, train
from arena.pong.players import GoodTeacher, HabitTeacher

CONFIG = PongConfig()


def test_good_teacher_returns_every_ball():
    report = evaluate(GoodTeacher(CONFIG), 3_000, CONFIG, seed=5)
    assert report.hit_rate == 1.0


def test_student_imitates_the_teacher_including_the_habit():
    teacher = HabitTeacher(CONFIG)
    trained = train(record(teacher, 8_000, CONFIG, seed=1), CONFIG)
    assert trained.test_accuracy > 0.9
    student_report = evaluate(trained.student, 3_000, CONFIG, seed=7)
    assert student_report.time_at_top > 0.4  # o vício (ficar no topo) foi copiado
    assert evaluate(GoodTeacher(CONFIG), 3_000, CONFIG, seed=7).time_at_top < 0.05
