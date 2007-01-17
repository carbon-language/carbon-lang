
#
# DUP test
#
FORWARD success;
FORWARD failure;
: phase3 0 = IF success ELSE failure ENDIF ;
: phase2 2 = IF phase3 ELSE failure ENDIF ;
: MAIN 0 1 2 SWAP 1 = IF phase2 ELSE failure ENDIF ;
