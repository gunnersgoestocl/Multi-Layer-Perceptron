CC = gcc # C compiler
AR = ar # archiver

all: src/adv_regression.c src/init.c
	make init.o
	make adv_regression.o
	make lib
	make demo
	make do

do: bin/demo
	bin/demo wine+quality/winequality-red.csv

demo: lib/libadv_regression.a
	$(CC) -o bin/demo -L./lib -ladv_regression

lib: src/init.o  src/adv_regression.o
	$(AR) rsv lib/libadv_regression.a src/init.o src/adv_regression.o

init.o: src/init.c
	$(CC) -I./include -c -o src/init.o src/init.c 

# prop.o: src/prop.c
# 	$(CC) -I./include -c -o src/prop.o src/prop.c

adv_regression.o: src/adv_regression.c
	$(CC) -I./include -c -o src/adv_regression.o src/adv_regression.c 

.PHONY: clean
clean:
	rm src/*.o lib/*.a bin/*