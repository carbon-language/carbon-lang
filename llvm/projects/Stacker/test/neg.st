#
# NEG test
#
FORWARD success;
FORWARD failure;
: MAIN 23 NEG -23 = IF success ELSE failure ENDIF ;
