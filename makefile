
SRCDIR=./src
INCLDIR=./src $(HOME)/local/include # change to your opencv include folder
LIBDIR=$(HOME)/local/lib # change to your opencv lib folder
LIBS=opencv_core opencv_highgui opencv_imgproc
CFLAGS=-Wall -O2 -fopenmp $(patsubst %,-I%,$(INCLDIR))
LDFLAGS=$(patsubst %,-L%,$(LIBDIR)) $(patsubst %,-l%,$(LIBS))
SOURCES=main.cpp BMS.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=BMS

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	g++ $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: $(SRCDIR)/%.cpp
	g++ -c $(CFLAGS) $< -o $@


clean: 
	rm -rf *o $(EXECUTABLE)