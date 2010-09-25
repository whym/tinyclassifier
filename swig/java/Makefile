TARGET=TinyClassifier
SO=lib$(TARGET).so
JAVAC=javac -source 1.6
JAVA=java
JAR=jar
CXX=c++
INCLUDE=$(JAVA_HOME)/include
PACKAGE=com/github/whym/tinyclassifier
SOURCES=$(PACKAGE)/*.java $(PACKAGE)/test/*.java
LIBS=-L../../lib
INC=-I$(INCLUDE) -I$(INCLUDE)/linux -I../../include

all: $(SO) $(TARGET).jar

$(SO): *_wrap.cxx
	$(CXX) -O3 -c -fpic $(TARGET)_wrap.cxx  $(INC)
	$(CXX) -shared -o $(SO) $(TARGET)_wrap.o $(LIBS)
$(TARGET).jar: $(SOURCES)
	$(JAVAC) $(SOURCES)
	$(JAR) cfv $(TARGET).jar $(subst .java,.class,$(SOURCES))

test: all
	./run_tests.sh

clean:
	rm -fr *.jar *.o *.so *.class $(PACKAGE)/*.class

%.class: %.java
	$(JAVAC) $<
