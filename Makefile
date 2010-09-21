all:
#	make -C lib
	make -C include
	make -C test
clean:
#	make -C lib $@
	make -C include $@
	make -C test $@
