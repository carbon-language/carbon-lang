set terminal png

##------- Plot small Date vs LOC ----
set output "running_loc.png"
set xlabel "Date" "TimesRoman,24"
set ylabel "Lines of Code" "TimesRoman,24"
set xdata time
set timefmt "%Y-%m-%d:"
set format x "%b %d, %Y"

## Various labels for the graph
set label "Removed\ndummy\nfunction" at "2003-07-30:", 150000

set size .75,.75
plot "running_loc.txt" using 1:2 title '', \
     "running_loc.txt" using 1:2 title "Date vs. Lines of Code" with lines

##------- Plot large Date vs LOC ----
set size 1.5,1.5
set output "running_loc_large.png"
plot "running_loc.txt" using 1:2 title '', \
     "running_loc.txt" using 1:2 title "Date vs. Lines of Code" with lines


# Delete all labels...
set nolabel


