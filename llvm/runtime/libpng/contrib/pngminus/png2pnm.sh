# -- grayscale
./png2pnm -noraw ../pngsuite/basn0g01.png basn0g01.pgm
./png2pnm -noraw ../pngsuite/basn0g02.png basn0g02.pgm
./png2pnm -noraw ../pngsuite/basn0g04.png basn0g04.pgm
./png2pnm -noraw ../pngsuite/basn0g08.png basn0g08.pgm
./png2pnm -noraw ../pngsuite/basn0g16.png basn0g16.pgm
# -- full-color
./png2pnm -noraw ../pngsuite/basn2c08.png basn2c08.ppm
./png2pnm -noraw ../pngsuite/basn2c16.png basn2c16.ppm
# -- palletted
./png2pnm -noraw ../pngsuite/basn3p01.png basn3p01.ppm
./png2pnm -noraw ../pngsuite/basn3p02.png basn3p02.ppm
./png2pnm -noraw ../pngsuite/basn3p04.png basn3p04.ppm
./png2pnm -noraw ../pngsuite/basn3p08.png basn3p08.ppm
# -- gray with alpha-channel
./png2pnm -noraw ../pngsuite/basn4a08.png basn4a08.pgm
./png2pnm -noraw ../pngsuite/basn4a16.png basn4a16.pgm
# -- color with alpha-channel
./png2pnm -noraw -alpha basn6a08.pgm ../pngsuite/basn6a08.png basn6a08.ppm
./png2pnm -noraw -alpha basn6a16.pgm ../pngsuite/basn6a16.png basn6a16.ppm
# -- grayscale
./png2pnm -raw ../pngsuite/basn0g01.png rawn0g01.pgm
./png2pnm -raw ../pngsuite/basn0g02.png rawn0g02.pgm
./png2pnm -raw ../pngsuite/basn0g04.png rawn0g04.pgm
./png2pnm -raw ../pngsuite/basn0g08.png rawn0g08.pgm
./png2pnm -raw ../pngsuite/basn0g16.png rawn0g16.pgm
# -- full-color
./png2pnm -raw ../pngsuite/basn2c08.png rawn2c08.ppm
./png2pnm -raw ../pngsuite/basn2c16.png rawn2c16.ppm
# -- palletted
./png2pnm -raw ../pngsuite/basn3p01.png rawn3p01.ppm
./png2pnm -raw ../pngsuite/basn3p02.png rawn3p02.ppm
./png2pnm -raw ../pngsuite/basn3p04.png rawn3p04.ppm
./png2pnm -raw ../pngsuite/basn3p08.png rawn3p08.ppm
# -- gray with alpha-channel
./png2pnm -raw ../pngsuite/basn4a08.png rawn4a08.pgm
./png2pnm -raw ../pngsuite/basn4a16.png rawn4a16.pgm
# -- color with alpha-channel
./png2pnm -noraw -alpha rawn6a08.pgm ../pngsuite/basn6a08.png rawn6a08.ppm
./png2pnm -noraw -alpha rawn6a16.pgm ../pngsuite/basn6a16.png rawn6a16.ppm

