all:
	g++ -std=c++14 -pedantic -Wall -Wextra -O3 -mavx2 -DNDEBUG main.cpp -isystem include
