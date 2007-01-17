#
# DUP test
#
FORWARD success;
FORWARD failure;
: phase3 1 = IF success ELSE failure ENDIF ;
: phase2 2 = IF phase3 ELSE failure ENDIF ;
: MAIN 1 2 DUP 2 = IF phase2 ELSE failure ENDIF ;
