from sklearn.linear_model import LogisticRegression
#from errortools.lofo_importance import LOFOImportance, plot_importance
from data.test_data import generate_test_data, generate_unstructured_test_data


def test_stuff():
    df = generate_test_data(1000)

    features = ["A", "B", "C", "D"]

    assert True==True, "Message if assert fails"