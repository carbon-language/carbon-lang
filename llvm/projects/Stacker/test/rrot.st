#
# RROT test
#
# Logic:  w1 w2 w3 -- w3 w1 w2 
#
FORWARD success;
FORWARD failure;
: try_0 0 = IF success ELSE failure ENDIF ;
: try_3 3 = IF try_0 ELSE failure ENDIF ;
: try_1 1 = IF try_3 ELSE failure ENDIF ;
: MAIN 0 1 2 3 RROT 2 = IF try_1 ELSE failure ENDIF ;

