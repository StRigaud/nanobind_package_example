// nanobind includes
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// C++ includes
#include <limits>
#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

struct Vertex
{
    double x;
    double y;
    double z;

    Vertex(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
    
    bool operator>(const Vertex &other) const {
        return (x + y + z) > (other.x + other.y + other.z);
    }
};

struct Edge
{
    nb::list vertices;

    Edge() = default;
    Edge(nb::list v) : vertices(v) {
        for (auto &&v : vertices) {
            if (!nb::isinstance<Vertex>(v)) {
                throw std::invalid_argument("Edge vertices must be of type Vertex");
            }
        }
        vertices.attr("sort")(nb::arg("reverse") = true);
    }

    bool add_vertex(const Vertex &v) {
        vertices.append(v);
        return true;
    }
    bool add_vertices(const nb::list &v) {
        for (auto &&vertex : v) {
            if (!nb::isinstance<Vertex>(vertex)) {
                throw std::invalid_argument("Edge vertices must be of type Vertex");
            }
            vertices.append(vertex);
        }
        return true;
    }
};

struct Cell
{    
    nb::list vertices;
    nb::list edges;
    nb::list neighbors;

    Cell(nb::list vertices, nb::list edges) : vertices(vertices), edges(edges) {}
    Cell(nb::list vertices, nb::list edges, nb::list neighbors) : vertices(vertices), edges(edges), neighbors(neighbors) {}
};


void parse_tissue(nb::ndarray<int, nb::ndim<3>, nb::c_contig, nb::device::cpu> image) {

    // Get buffer info
    auto ptr = image.data();
    auto ndim = image.ndim();
    auto size = image.size();

    auto v = image.view();

    for (size_t i = 0; i < v.shape(0); ++i) // Important; use 'v' instead of 'arg' everywhere in loop
        for (size_t j = 0; j < v.shape(1); ++j)
            for (size_t k = 0; k < v.shape(2); ++k)
                v(i, j, k);
 

}









// PYBIND11_MODULE(tissue, m) {
NB_MODULE(tissue, m) {
    m.doc() = "nanobind example plugin"; // optional module docstring

    m.def("parse_tissue", &parse_tissue, "Add arrays", "image"_a);

    nb::class_<Vertex>(m, "Vertex")
        .def(nb::init<size_t, size_t, size_t>())
        .def("__gt__", &Vertex::operator>)
        .def("__repr__", [](const Vertex &v) {
            return "<Vertex [" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "]>";
        })
        .def("__str__", [](const Vertex &v) {
            return "[" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "]";
        })
        .def_rw("x", &Vertex::x)
        .def_rw("y", &Vertex::y)
        .def_rw("z", &Vertex::z);

    nb::class_<Edge>(m, "Edge")
        .def(nb::init<nb::list>())
        .def_rw("vertices", &Edge::vertices);

    nb::class_<Cell>(m, "Cell")
        .def(nb::init<nb::list, nb::list>())
        .def(nb::init<nb::list, nb::list, nb::list>())
        .def_rw("vertices", &Cell::vertices)
        .def_rw("edges", &Cell::edges)
        .def_rw("neighbors", &Cell::neighbors);
}

