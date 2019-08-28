# Recommender-Systems-related-to-Reviews-in-Amazon

Summary: 

I finish two different goals. First, my model should predict whether the user would purchase the item. In other word, based on this model, the system can give the customer what he might interested. Second, the other model should give the product category by given "customer review". 

Detail:

Purchase-predict:
In this part, I use Jaccard similarity method and the popular method to predict the user will buy the item or not.
The Jaccard similarity method is first use a “dict” to store all the categories of all users buy(the key is “user”). When getting a new “user, item”, I use Jaccard similarity to compute the most similar 1600 users to this user. Then, if this item was bought by these users, return true.
Furthermore, in predicting this item will be bought by this user, the if condition is: Jaccard similarity function return true OR this item is popular. The popular method is same as the popular predict baseline which professor gave to us.

Categories-predict:
First I take advantage of my previous project--get the “common words”. Then import SVM for prediction. For the feature function, I build up feature vector out of 2000 words. I build a decision function. This make the business to predict in the category which has the highest confidence. This is not enough for a high accuracy. Hence, I add more feature in the decision function. Be specific, decide the categoryID is in the users’ categoryID or not. If I use the common words and the new decision feature, the accuracy is around 0.84.
