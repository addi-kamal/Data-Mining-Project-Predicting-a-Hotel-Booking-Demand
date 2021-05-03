# I. Dataset: 

1.	Data description  :

This data describes two datasets with hotel demand data. One of the hotels (H1) is a resort hotel and the other is a city hotel (H2). Both datasets share the same structure, with 31 variables describing the 40,060 observations of H1 and 79,330 observations of H2. Each observation represents a hotel booking.  Since this is hotel real data, all data elements pertaining hotel or costumer identification were deleted. 

2.	Data Dictionary :



| **Variable**        | **Class**      | **Description**  |
| ------|-----|-----|
|hotel | 	character |	Hotel (H1 = Resort Hotel or H2 = City Hotel) |
|is_canceled |	double 	| Value indicating if the booking was canceled (1) or not (0) |
| lead_time |	double |	Number of days that elapsed between the entering date of the booking into the PMS and the arrival date |
|arrival_date_year |	double |	Year of arrival date |
|	arrival_date_month |		character 	|	Month of arrival date |	
|	arrival_date_week_nu mber | 	double | 	Week number of year for arrival date |	
|	arrival_date_day_of_month |	double  |	Day of arrival date |	
|	stays_in_weekend_ni ghts |		double | 	Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel |	
|	stays_in_week_nights |		double |		Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel |	
|	adults |		double |		Number of adults |	
|	children |		double | 	Number of children |	
|	babies |		double | 	Number of babies |	
|	meal |		character |		Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; |	
|	country  |		character  |		Country of origin. Categories are represented in the ISO 3155– 3:2013 format  |	
|	market_segment  |		character  |		Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”  |	
|	distribution_channel  |		character  |		Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”  |	
|	is_repeated_guest  |		double  |		Value indicating if the booking name was from a repeated guest (1) or not (0)  |	
|	previous_cancellation  |		double  |		Number of previous bookings that were cancelled by the customer prior to the current booking  |	
|	previous_bookings_n ot_canceled  |		double  |		Number of previous bookings not cancelled by the customer prior to the current booking  |	
|	reserved_room_type  |		character  |		Code of room type reserved. Code is presented instead of designation for anonymity reasons  |	
|	assigned_room_type  |		character  |		Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons  |	
|	booking_changes |double |	Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation |
|	deposit_type  |		character | 	Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.  |	
|	agent 	 |	character  |		ID of the travel agency that made the booking  |	
|	company  |	character |	ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons |
|	days_in_waiting_list  |	double  |	Number of days the booking was in the waiting list before it was confirmed to the customer  |	
|	customer_type  |		character  |		Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking  |	
|	adr  |		double | 	Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights  |	
|	required_car_parking _spaces  |		double  |	 	Number of car parking spaces required by the customer  |	
 |	total_of_special_requests  |	double 	 |	Number of special requests made by the customer (e.g. twin bed or high floor)  |	
 |	reservation_status  |		character  |		Reservation last status, assuming one of three categories: Canceled – booking was canceled by the customer; Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did inform the hotel of the reason why  |	
 |	reservation_status_date  |	double | 	Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel  |	
 
II. Data preprocessing :

1.	Verify missing values 
2.	Drop rows having missing values except for variables like Agent or Company, “NULL” is presented as one of the categories. 
3.	Change arrival year, month and day feature to datetime format called arrival_date. 
4.	Verify that the timestamp of the variable reservation_status_date must occur after or at the same date as the input variable arrival_date 
5.	Propose a preprocessing to be made on this dataset. 

III. Exploratory data analysis: 

1.	Create dataset summary statistics – Date variables. 
2.	Create dataset summary statistics – Categorical variables. 
3.	Create dataset summary statistics – Integer and numeric variables. 
4.	Check the distribution of hotel type for cancellation 
5.	Plot distribution of cancellation and Number of Adults 
6.	Taking in consideration the characteristics of the variables included in this dataset propose two possible modeling this dataset can have an important role for research and education in revenue management (i.e define two possible target variables and the purpose of each analysis) 

IV. Modeling :

Like any business, hotels are also looking to gain profit. Propose a model that predicts if the booking is likely to be canceled which could be a good indication for hotels, as they may prefer to accept the lower risk bookings first. Choose two data mining technics and compare their performances. 
 
 
