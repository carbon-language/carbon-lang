#
# SELECT test
#
FORWARD success;
FORWARD failure;
: try_99 99 = IF success ELSE failure ENDIF ;
: MAIN 99 10 9 8 7 6 5 4 3 2 1 10 5 SELECT 5 = IF try_99 ELSE failure ENDIF ;
