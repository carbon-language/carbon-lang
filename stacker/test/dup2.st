#
# DUP2 test
#
FORWARD success;
FORWARD failure;
: phase2 1 = IF success ELSE failure ENDIF ;
: MAIN 1 2 DUP2 2 = IF phase2 ELSE failure ENDIF ;
