#
# PICK test
#
# Logic:  // x0 ... Xn n -- x0 ... Xn x0
#
FORWARD success;
FORWARD failure;
: next 10 = IF success ELSE failure ENDIF ;
: MAIN 0 1 2 3 4 5 6 7 8 9 10 5 PICK 5 = IF next ELSE failure ENDIF ;
