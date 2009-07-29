all:
#	make -C lib
	make -C include
	make -C test
clean:
#	make -C lib clean
	make -C test clean
