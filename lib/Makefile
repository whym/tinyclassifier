libtinyclassifier.so: perceptron.o
	g++ -fPIC -I../include -shared -Wl,-soname,$@ -o $@ $<
clean:
	-rm libtinyclassifier.so perceptron.o
%.o: %.cc
	g++ -c -I../include -o $@ $<
