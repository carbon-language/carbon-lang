#
# MEMORY test
#
FORWARD success;
FORWARD failure;
: try_free FREE ;
: read_space >s ;
: write_space 0 72 PUT 1 69 PUT 2 76 PUT 3 76 PUT 4 79 PUT ;
: try_malloc 64 MALLOC ;
: MAIN try_malloc write_space read_space try_free " - " >s success ;
