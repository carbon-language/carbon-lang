#
# OVER test
#
FORWARD success;
FORWARD failure;
: phase4 0 = IF success ELSE failure ENDIF ;
: phase3 2 = IF phase4 ELSE failure ENDIF ;
: phase2 1 = IF phase3 ELSE failure ENDIF ;
: MAIN 0 2 1 OVER 2 = IF phase2 ELSE failure ENDIF ;
