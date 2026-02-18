So, this script look at the previous iteration of an aliquot sequence.
To calculate one, you need to find a number such as S(n)-n give you the number you are looking the antecedent for.
S(n) is the sum of all the divisor of the number.
for example,  you look at the antecedent of 
1066, it will find 1310, because its factorisation is 2*5*131, wich is equal to 1+2+5+131=139, +10(2*5)+262(2*131)+655(5*131)=927, and add the number itself 1310, 
that give an S(n) of 2376, which when you substract 1310, give you 1066.
The problem is that there is often more than one, as 1066 has 1310, 1412 and 2126 as antecedent.

20 sec to find an antecedent of a 100e9 number

