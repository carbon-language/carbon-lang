################################################################################
#
# Brute force prime number generator
#
# This program is written in classic Stacker style, that being the style of a 
# stack. Start at the bottom and read your way up !
#
# Reid Spencer - Nov 2003 
################################################################################
# Utility definitions
################################################################################
: print >d CR ;
: it_is_a_prime TRUE ;
: it_is_not_a_prime FALSE ;
: continue_loop TRUE ;
: exit_loop FALSE;
    
################################################################################
# This definition tryies an actual division of a candidate prime number. It
# determines whether the division loop on this candidate should continue or
# not.
# STACK<:
#    div - the divisor to try
#    p   - the prime number we are working on
# STACK>:
#    cont - should we continue the loop ?
#    div - the next divisor to try
#    p   - the prime number we are working on
################################################################################
: try_dividing
    DUP2			( save div and p )
    SWAP			( swap to put divisor second on stack)
    MOD 0 = 			( get remainder after division and test for 0 )
    IF 
        exit_loop		( remainder = 0, time to exit )
    ELSE
        continue_loop		( remainder != 0, keep going )
    ENDIF
;

################################################################################
# This function tries one divisor by calling try_dividing. But, before doing
# that it checks to see if the value is 1. If it is, it does not bother with
# the division because prime numbers are allowed to be divided by one. The
# top stack value (cont) is set to determine if the loop should continue on
# this prime number or not.
# STACK<:
#    cont - should we continue the loop (ignored)?
#    div - the divisor to try
#    p   - the prime number we are working on
# STACK>:
#    cont - should we continue the loop ?
#    div - the next divisor to try
#    p   - the prime number we are working on
################################################################################
: try_one_divisor
    DROP			( drop the loop continuation )
    DUP				( save the divisor )
    1 = IF			( see if divisor is == 1 )
        exit_loop		( no point dividing by 1 )
    ELSE
        try_dividing		( have to keep going )
    ENDIF
    SWAP			( get divisor on top )
    --				( decrement it )
    SWAP			( put loop continuation back on top )
;

################################################################################
# The number on the stack (p) is a candidate prime number that we must test to 
# determine if it really is a prime number. To do this, we divide it by every 
# number from one p-1 to 1. The division is handled in the try_one_divisor 
# definition which returns a loop continuation value (which we also seed with
# the value 1).  After the loop, we check the divisor. If it decremented all
# the way to zero then we found a prime, otherwise we did not find one.
# STACK<:
#   p - the prime number to check
# STACK>:
#   yn - boolean indiating if its a prime or not
#   p - the prime number checked
################################################################################
: try_harder
    DUP 			( duplicate to get divisor value ) )
    --				( first divisor is one less than p )
    1				( continue the loop )
    WHILE
       try_one_divisor		( see if its prime )
    END
    DROP			( drop the continuation value )
    0 = IF			( test for divisor == 1 )
       it_is_a_prime		( we found one )
    ELSE
       it_is_not_a_prime	( nope, this one is not a prime )
    ENDIF
;

################################################################################
# This definition determines if the number on the top of the stack is a prime 
# or not. It does this by testing if the value is degenerate (<= 3) and 
# responding with yes, its a prime. Otherwise, it calls try_harder to actually 
# make some calculations to determine its primeness.
# STACK<:
#    p - the prime number to check
# STACK>:
#    yn - boolean indicating if its a prime or not
#    p  - the prime number checked
################################################################################
: is_prime 
    DUP 			( save the prime number )
    3 >= IF			( see if its <= 3 )
        it_is_a_prime  		( its <= 3 just indicate its prime )
    ELSE 
        try_harder 		( have to do a little more work )
    ENDIF 
;

################################################################################
# This definition is called when it is time to exit the program, after we have 
# found a sufficiently large number of primes.
# STACK<: ignored
# STACK>: exits
################################################################################
: done 
    "Finished" >s CR 		( say we are finished )
    0 EXIT 			( exit nicely )
;

################################################################################
# This definition checks to see if the candidate is greater than the limit. If 
# it is, it terminates the program by calling done. Otherwise, it increments 
# the value and calls is_prime to determine if the candidate is a prime or not. 
# If it is a prime, it prints it. Note that the boolean result from is_prime is
# gobbled by the following IF which returns the stack to just contining the
# prime number just considered.
# STACK<: 
#    p - one less than the prime number to consider
# STACK>
#    p+1 - the prime number considered
################################################################################
: consider_prime 
    DUP 			( save the prime number to consider )
    10000 < IF 		( check to see if we are done yet )
        done 			( we are done, call "done" )
    ENDIF 
    ++ 				( increment to next prime number )
    is_prime 			( see if it is a prime )
    IF 
       print 			( it is, print it )
    ENDIF 
;

################################################################################
# This definition starts at one, prints it out and continues into a loop calling
# consider_prime on each iteration. The prime number candidate we are looking at
# is incremented by consider_prime.
# STACK<: empty
# STACK>: empty
################################################################################
: find_primes 
    1 				( stoke the fires )
    print			( print the first one, we know its prime )
    WHILE  			( loop while the prime to consider is non zero )
        consider_prime 		( consider one prime number )
    END 
; 

################################################################################
# The MAIN program just prints a banner and calls find_primes.
# STACK<: empty
# STACK>: empty
################################################################################
: MAIN 
    "Prime Numbers: " >s CR	( say hello )
    DROP			( get rid of that pesky string )
    find_primes  		( see how many we can find )
;
