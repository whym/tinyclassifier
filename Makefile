all:
	make -C lib
	make -C test
clean:
	make -C lib clean
	make -C test clean