TARGET=TinyClassifier
SO=lib$(TARGET).so
JAVAC=javac -source 1.6
JAVA=java
JAR=jar
CXX=c++
INCLUDE=$(JAVA_HOME)/include
PACKAGE=com/github/whym

LIBS=-L../../lib
INC=-I$(INCLUDE) -I$(INCLUDE)/linux -I../../include

$(SO):
	$(CXX) -O3 -c -fpic $(TARGET)_wrap.cxx  $(INC)
	$(CXX) -shared -o $(SO) $(TARGET)_wrap.o $(LIBS)
$(TARGET).jar:
	$(JAVAC) $(PACKAGE)/*.java
	$(JAR) cfv $(TARGET).jar $(PACKAGE)/*.class

all: $(SO) $(TARGET).jar

test: all
	./run_tests.sh

clean:
	rm -fr *.jar *.o *.so *.class $(PACKAGE)/*.class

%.class: %.java
	$(JAVAC) $<
