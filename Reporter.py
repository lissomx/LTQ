from sklearn.metrics import confusion_matrix, classification_report

Runned = {}
def runNtime(fun, n=1, tag=''):
    if Runned.setdefault(tag, 0) < n:
        fun()
    Runned[tag] += 1

Values = {}

def record(value, group=''):
    global Values
    Values_ = Values.setdefault(group, [])
    Values_.append(value)

def report(group='', mthd='mean'):
    global Values
    Values_ = Values[group]
    if mthd == 'mean':
        value = sum(Values_) / len(Values_)
    Values[group] = []
    return value



Prediction = {}
Target = {}

def srecord(prediction, target, group=''):
    global Prediction
    global Target
    Prediction_ = Prediction.setdefault(group, [])
    Target_ = Target.setdefault(group, [])
    Prediction_ += prediction.tolist()
    Target_ += target.tolist()

def sreport(group=''):
    # Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
    global Prediction
    global Target
    Prediction_ = Prediction[group]
    Target_ = Target[group]
    r = classification_report(Target_, Prediction_, digits=4)
    x = confusion_matrix(Target_, Prediction_)
    Prediction[group] = []
    Target[group] = []
    return r+'\n'+str(x)