"""
Check that model can be imported
"""

from whalevad import whalevad
from whalevad import WhaleVADClassifier

model = whalevad(weights="DEFAULT")

classifier, transform = model
assert isinstance(classifier, WhaleVADClassifier)


# TODO implement an end to end test
