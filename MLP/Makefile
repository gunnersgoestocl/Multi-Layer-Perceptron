CC = gcc # C compiler
AR = ar # archiver
TODAY   = $(shell date "+%Y%m%d%H%M%S")

all: src/adv_regression.c src/init.c
	make init.o
	make adv_regression.o
	make lib
	make demo
	make sample

do: bin/demo
	bin/demo wine+quality/winequality-red.csv log/demo.txt

sample: bin/demo
	bin/demo sample.txt log/sample_output_$(TODAY).txt log/sample_output_graph_$(TODAY)/sample_output_graph_$(TODAY)

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