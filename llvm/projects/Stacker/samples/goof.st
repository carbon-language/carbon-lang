#
# goof
#
: print_one
    --
    SWAP
    >s
    DROP
;
: print_it
    WHILE 
        print_one
    END
;
    
: MAIN
    "MICKEY: I said she was f'in goofy!"
    "MICKEY: I didn't say she was insane." 
    "JUDGE: Yet you provide no evidence of this and I do not concur."
    "JUDGE: In your pleadings you claim that Mini Mouse is insane."
    "MICKEY: Well, what do you mean, your honor?"
    "JUDGE: Mr. Mouse, I find your grounds for divorce insufficient. "
    6
    print_it
;
