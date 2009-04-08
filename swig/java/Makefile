TARGET=TinyClassifier
JAVAC=javac -source 1.5 
JAVA=java
JAR=jar
CXX=c++
INCLUDE=$(JAVA_HOME)/include

PACKAGE=com/github/whym

LIBS=-L../../lib
INC=-I$(INCLUDE) -I$(INCLUDE)/linux -I../../include

all:
	$(CXX) -O3 -c -fpic $(TARGET)_wrap.cxx  $(INC)
	$(CXX) -shared -o lib$(TARGET).so $(TARGET)_wrap.o $(LIBS)
	$(JAVAC) $(PACKAGE)/*.java
	$(JAR) cfv $(TARGET).jar $(PACKAGE)/*.class

test: test.class
	env LD_LIBRARY_PATH=. $(JAVA) test

clean:
	rm -fr *.jar *.o *.so *.class $(PACKAGE)/*.class

%.class: %.java
	$(JAVAC) $<
