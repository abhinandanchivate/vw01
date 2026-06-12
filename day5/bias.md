Real-Time Example 3: Email Spam Detection

A spam model may learn:

If email contains “offer”, mark it as spam.

This is high bias.

Because genuine emails can also contain “offer”.

Example:

“Your salary offer letter is attached.”

This is not spam, but the model may classify it as spam.

How to Identify Bias in ML
Training Accuracy	Testing Accuracy	Meaning
Low	Low	High Bias / Underfitting
High	Low	High Variance / Overfitting
High	High	Good Fit

Example:

Model	Training Accuracy	Testing Accuracy	Result
Very simple model	65%	63%	High Bias
Very complex model	99%	72%	High Variance
Balanced model	88%	85%	Good Fit
How Bias is Reduced
Problem	Solution
Model is too simple	Use a more powerful model
Not enough features	Add important features
Wrong algorithm	Try another algorithm
Very shallow tree	Increase max_depth
Poor feature engineering	Create better input features
Decision Tree Example
Model	Bias
max_depth=1	High bias
max_depth=3	Balanced
max_depth=None	Low bias, but overfitting risk
Simple Meaning

max_depth=1 means the tree can ask only one main question.

Example:

Is mileage greater than 50,000?

That is too simple for a real automobile risk prediction problem.

So it may have high bias.
