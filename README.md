# Project_two_zillow


# Project Description

predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.


# Project Goal

- To create a model that is capable of predicting tax assessed values more than 50 percent of the time.


# Initial Thoughts

My initial hypothesis is that I will be able to create a model that beats 50% accuracy with the features we have been going over in class


# The Plan

- Aquire zillow data from CodeUp Database


- prepare data
    - checked that the column datatypes were appropriate
    - droped the unnamed: 0 column
    - created a boolean mask to find calculated square feet values under 25000
    - cut the top 5% of taxvaluedollar count
    - dropped the na values
    - converted calculatedfinishedsquarefeet, fips and yearbuilt to int's
    - Split data into train, validate and test (approx. 60/20/20)
    
- Explore the data in search of drivers of tax assessed values

    - Answer the following questions
        
        - Does year built affect tax assessed values
        - Does bedroom count affect tax assessed values
        - Does bathroom count affect tax assessed values
        - Does year calculated finished square feet tax assessed values
        
    - Develop a Model to predict future values of tax assessed values
        
        - Use the drivers identified in explore to build a predictive model of different types
        - Evaluate models on train and validate data
        - Search for the best model based on highest accuracy
        - Evaluate the best model on test data
        
- Draw conclusions


# Data Dictionary


<table><tr><td class="border_l border_r border_t border_b selected"><div class="wrap"><div class="" contenteditable="false" style="margin: 10px 5px;"><p><span>taxvaluedollarcnt</span></p></div></div></td><td class="border_l border_r border_t border_b selected" style="border-right-width: 1px; border-right-color: inherit;"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>Tax assessed values of Single Family Properties</span></p><p><span>that had a transaction during 2017.</span></p></div></div></td></tr><tr><td class="border_l border_r border_t border_b selected"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>bedroomcnt</span></p></div></div></td><td class="border_l border_r border_t border_b selected" style="border-right-width: 1px; border-right-color: inherit;"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>The amount of bedrooms of Single Family Properties</span></p><p><span>that had a transaction during 2017</span></p></div></div></td></tr><tr><td class="border_l border_r border_t border_b selected"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>calculatedfinishedsquarefeet</span></p></div></div></td><td class="border_l border_r border_t border_b selected" style="border-right-width: 1px; border-right-color: inherit;"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>The amount of Square feet of Single Family Properties</span></p><p><span>that had a transaction during 2017</span></p></div></div></td></tr><tr><td class="border_l border_r border_t border_b selected"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>bathroomcnt</span></p></div></div></td><td class="border_l border_r border_t border_b selected" style="border-right-width: 1px; border-right-color: inherit;"><div class="wrap"><div style="margin: 10px 5px;" class="" contenteditable="false"><p><span>The amount of bathrooms of Single Family Properties</span></p><p><span>that had a transaction during 2017</span></p></div></div></td></tr><tr><td class="border_l border_r border_t border_b selected"><div class="wrap"><div class="" contenteditable="false" style="margin: 10px 5px;"><p><span>yearbuilt</span></p></div></div></td><td class="border_l border_r border_t border_b selected"><div class="wrap"><div class="" contenteditable="false" style="margin: 10px 5px;"><p><span>The year built of Single Family Properties</span></p><p><span>that had a transaction during 2017</span></p></div></div></td></tr></table>


# Steps to Reproduce

- clone this repo
- Aquire the data from CodeUp Database using the final_report file
- Run final_report


# Takeaways and Conclusions

- calculated finished square feet affect tax value dollar count
- bathroom count affect tax value dollar count
- bedroom count affect tax value dollar count
- year built affect tax value dollar count
    - The average of tax value dollar count has went up over the years
    - The average of tax value dollar count is higher the more bedrooms a home posseses
    - The average of tax value dollar count is higher the more square feet a home posseses
    - The average of tax value dollar count is higher bathrooms a home posseses
    
    
# Recommendations

- To not continue on with any of the models used in the modeling phase
- Look for more things that affect tax value dollar count
- Break the catagories up by location subcatagories
- Imput new features into the modeling tests
