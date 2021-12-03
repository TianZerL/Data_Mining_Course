from dataset import (
    Car,
    Heart,
    HeartMore,
    Iris,
    Mushroom,
    MushroomOneHotEncode,
    Wine,
    WineNormalized,
)
from score import cal_scores, print_scores

print_scores("Iris", cal_scores(Iris()))
print_scores("Wine", cal_scores(Wine()))
print_scores("Wine归一化", cal_scores(WineNormalized()))
print_scores("Car Evaluation", cal_scores(Car()))
print_scores("心脏病", cal_scores(Heart()))
print_scores("心脏病增加数据", cal_scores(HeartMore()))
print_scores("Mushroom OneHotEncode", cal_scores(MushroomOneHotEncode()))
print_scores("Mushroom", cal_scores(Mushroom()))
