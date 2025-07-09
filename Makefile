CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -I/opt/homebrew/include -Wall -Wextra
LDFLAGS = -L/opt/homebrew/lib -lGLEW -lglfw -framework OpenGL -framework OpenCL

SRC_DIR = src
OBJ_DIR = obj
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

EXEC = particle_system

all: $(EXEC)

$(EXEC): $(OBJ_FILES)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(EXEC)

re: clean all

.PHONY: all clean