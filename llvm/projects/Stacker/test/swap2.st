#
# SWAP2 test
#
FORWARD success;
FORWARD failure;
: try_0 0 = IF success ELSE failure ENDIF ;
: try_3 3 = IF try_0 ELSE failure ENDIF ;
: try_4 4 = IF try_3 ELSE failure ENDIF ;
: try_1 1 = IF try_4 ELSE failure ENDIF ;
: MAIN 0 1 2 3 4 SWAP2 2 = IF try_1 ELSE failure ENDIF ;
