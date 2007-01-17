#
# NIP2 test
#
FORWARD success;
FORWARD failure;
: test_0 0 = IF success ELSE failure ENDIF ;
: test_3 3 = IF test_0 ELSE failure ENDIF ;
: test_4 4 = IF test_3 ELSE failure ENDIF ;
: MAIN 0 1 2 3 4 NIP2 test_4 ; 
