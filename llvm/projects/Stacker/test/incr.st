#
# INCR test
#
FORWARD success;
FORWARD failure;
: MAIN 7 ++ 8 = IF success ELSE failure ENDIF ;
