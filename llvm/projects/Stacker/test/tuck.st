#
# TUCK test
#
# Logic:  // w1 w2 -- w2 w1 w2
#
FORWARD success;
FORWARD failure;
: try_0 0 = IF success ELSE failure ENDIF ;
: try_2 2 = IF try_0 ELSE failure ENDIF ;
: try_1 1 = IF try_2 ELSE failure ENDIF ;
: MAIN 0 1 2 TUCK 2 = IF try_1 ELSE failure ENDIF ;
