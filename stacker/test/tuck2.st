#
# TUCK2 test
#
# Logic:  // w1 w2 w3 w4 -- w3 w4 w1 w2 w3 w4
#
FORWARD success;
FORWARD failure;
: try_0 0 = IF success ELSE failure ENDIF ;
: try_3b 3 = IF try_0 ELSE failure ENDIF ;
: try_4 4 = IF try_3b ELSE failure ENDIF ;
: try_1 1 = IF try_4 ELSE failure ENDIF ;
: try_2 2 = IF try_1 ELSE failure ENDIF ;
: try_3a 3 = IF try_2 ELSE failure ENDIF ;
: MAIN 0 1 2 3 4 TUCK2 4 = IF try_3a ELSE failure ENDIF ;
