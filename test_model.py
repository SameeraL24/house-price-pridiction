import House_Price_Pridiction as hp

#Simple prediction function
def predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom,basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
    

    mainroad = 1 if mainroad == 'yes' else 0
    guestroom = 1 if guestroom == 'yes' else 0
    basement = 1 if basement == 'yes' else 0
    hotwaterheating = 1 if hotwaterheating == 'yes' else 0
    airconditioning = 1 if airconditioning == 'yes' else 0
    prefarea = 1 if prefarea == 'yes' else 0
    furnishingstatus = 0 if furnishingstatus == 'unfurnished' else 1 if furnishingstatus == 'semi-furnished' else 2
    
    features = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]]
    
    price = hp.modelpredict(features)
    return price

# Test the function
a= int(input("enter area of house: "))
b= int(input("enter bedrooms of house: "))
bathrooms =int(input("enter bathrooms of house: "))
s = int(input("enter how meny storys of house is: "))
m = input("enter how meny bedrooms are there: ")
g = input("guestroom are there or not :")
basement = input("basement are there or not :")
hotwaterheating= input("hotwaterheating are there or not :")
airconditioning = input ("airconditioning rooms are there or not")
parking=int(input ("airconditioning rooms are there or not"))
prefarea=input("enter how meny bedrooms are there: ")
furnishingstatus=input ("airconditioning rooms are there or not : ")


test_price = predict_house_price(a, b, bathrooms,s, m, g, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus)
print(f"\n Predicted Price for test house: {test_price:.2f}")