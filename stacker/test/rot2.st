#
# ROT2 test
#
# Logic:  // w1 w2 w3 w4 w5 w6 -- w3 w4 w5 w6 w1 w2
#
FORWARD success;
FORWARD failure;
: try_0 0 = IF success ELSE failure ENDIF ;
: try_3 3 = IF try_0 ELSE failure ENDIF ;
: try_4 4 = IF try_3 ELSE failure ENDIF ;
: try_5 5 = IF try_4 ELSE failure ENDIF ;
: try_6 6 = IF try_5 ELSE failure ENDIF ;
: try_1 1 = IF try_6 ELSE failure ENDIF ;
: MAIN 0 1 2 3 4 5 6 ROT2 2 = IF try_1 ELSE failure ENDIF ;
