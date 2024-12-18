// nanobind includes
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

// C++ includes
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;

// Define Edge and Cell class so that we can use it in Vertex class
struct Edge;
struct Cell;
struct Vertex;

struct Vertex
{
    double x;
    double y;
    double z;

    nb::list edges;
    nb::list cells;

    Vertex(double x, double y, double z) : x(x), y(y), z(z) {}

    bool operator<(const Vertex& other) const {
        return x+ y+z < other.x+ other.y+ other.z;
    }
    bool operator>(const Vertex& other) const {
        return x+ y+z > other.x+ other.y+ other.z;
    }

    std::string to_string() const  {
        return "v(" + std::to_string(z) + ", " + std::to_string(y) + ", " + std::to_string(x) + ")";
    }
};

struct Edge
{
    nb::list vertices; 
    nb::list cells;
    nb::tuple mid_point;

    Edge(nb::list v) : vertices(v) {
        for (auto &&v : vertices) {
            if (!nb::isinstance<Vertex>(v)) {
                throw std::invalid_argument("Edge vertices must be of type Vertex");
            }
        }
        vertices.attr("sort")(nb::arg("reverse") = true);
    }

    void update_mid_point()
    {
        double x = 0.;
        double y = 0.;
        double z = 0.;
        for (auto &&v : vertices) {
            x += (double) nb::cast<Vertex&>(v).x;
            y += (double) nb::cast<Vertex&>(v).y;
            z += (double) nb::cast<Vertex&>(v).z;
        }
        x /= (double) nb::len(vertices);
        y /= (double) nb::len(vertices);
        z /= (double) nb::len(vertices);
        mid_point = nb::make_tuple(z, y, x);
    }


    std::string to_string() const {
        return "e[ a=" + nb::cast<Vertex>(vertices[0]).to_string() + ", b=" + nb::cast<Vertex>(vertices[1]).to_string() + " ]";
    }
};

struct Cell
{    
    size_t id;
    double centroid_x;
    double centroid_y;
    double centroid_z;

    nb::tuple centroid;

    nb::list vertices;
    nb::list edges;
    nb::list neighbors;

    Cell(size_t id) : id(id) {}

    void update_centroid_from_vertices()
    {
        centroid_x = 0.;
        centroid_y = 0.;
        centroid_z = 0.;
        for (auto &&v : vertices) {
            centroid_x += (double) nb::cast<Vertex&>(v).x;
            centroid_y += (double) nb::cast<Vertex&>(v).y;
            centroid_z += (double) nb::cast<Vertex&>(v).z;
        }
        centroid_x /= (double) nb::len(vertices);
        centroid_y /= (double) nb::len(vertices);
        centroid_z /= (double) nb::len(vertices);
        centroid = nb::make_tuple(centroid_z, centroid_y, centroid_x);
    }

    void update_neighbors_from_vertices()
    {
        nb::list candidates;
        for (auto &&v : vertices) {
            for (auto &&c : nb::cast<Vertex&>(v).cells) {
                if (nb::cast<size_t>(c) == id)  {
                    continue;
                }
                candidates.append(nb::cast<size_t>(c));
            }
        }
        neighbors = nb::list(nb::set(candidates));
    }

    std::string to_string() const  {
        std::string pre = "Cell [\n";
        std::string post = "]\n";

        std::string print_id = "\tid: " + std::to_string(id) + ",\n";

        std::string print_nei = "\tneighbors: [";
        for (auto &&n : neighbors) {
            print_nei += std::to_string(nb::cast<size_t>(n)) + ", ";
        }
        print_nei += "],\n";

        std::string print_edges = "\tedges: [\n";
        for (auto &&e : edges) {
            print_edges += "\t\t" + nb::cast<Edge&>(e).to_string() + ",\n";
        }
        print_edges += "\t\t],\n";

        std::string to_print = pre + print_id + print_nei + print_edges + post;
        return to_print;
    }
};


std::vector<size_t> check_neigbors(const nb::ndarray<size_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>& image, size_t z, size_t y, size_t x) {
    auto arr_view = image.view();

    std::vector<size_t> values;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {

                auto cz = z + i;
                auto cy = y + j;
                auto cx = x + k;

                if (cx < 0 || cx >= arr_view.shape(2) ||
                    cy < 0 || cy >= arr_view.shape(1) ||
                    cz < 0 || cz >= arr_view.shape(0))
                    continue;

                auto value = arr_view(cz, cy, cx);
                if (value == 0)
                    continue;                

                values.push_back(value);
            }
        }
    }

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}


std::vector<Vertex> parse_vertices(const nb::ndarray<size_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>& image) {
    std::vector<Vertex> vertices;

    auto arr_view = image.view();
    for (size_t i = 0; i < arr_view.shape(0); ++i) {
        for (size_t j = 0; j < arr_view.shape(1); ++j) {
            for (size_t k = 0; k < arr_view.shape(2); ++k) {
                bool border = false;

                if (arr_view(i, j, k) != 0)
                    continue;

                // check if i,j,k is on the border for dimension bigger than 1
                for (auto l = 0; l < arr_view.ndim(); ++l) {
                    if (arr_view.shape(l) == 1)
                        continue;
                    
                    if (l == 0 && (i == 0 || i == arr_view.shape(0)-1)) {
                        border = true;
                        break;
                    }
                    if (l == 1 && (j == 0 || j == arr_view.shape(1)-1)) {
                        border = true;
                        break;
                    }
                    if (l == 2 && (k == 0 || k == arr_view.shape(2)-1)) {
                        border = true;
                        break;
                    }
                }
                
                auto neighbor_list = check_neigbors(image, i, j, k);
                if(neighbor_list.size() > 2 || border) {
                    auto v = Vertex(k, j, i);
                    for (auto &&n : neighbor_list) {
                        v.cells.append(n);
                    }
                    vertices.push_back(v);
                }
            }
        }
    }

    return vertices;
}


nb::list rebuild_tissue(const nb::ndarray<size_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>& image) 
{
    auto vertices = parse_vertices(image);

    // get the number max of cells
    size_t max_label = 0;
    for (auto &&v : vertices) {
        for (auto &&c : v.cells) {
            max_label = std::max(max_label, nb::cast<size_t>(c));
        }
    }

    // std::vector<Cell> cells;
    nb::list cells;
    // cells.reserve(max_label);
    for (size_t i = 0; i < max_label; ++i) {
        cells.append(Cell(i+1));
        
    }

    for (auto ite1 = 0; ite1 < vertices.size(); ++ite1) {\
        for (auto ite2 = ite1+1; ite2 < vertices.size(); ++ite2) {

            auto v1 = vertices[ite1];
            auto v2 = vertices[ite2];
            
            nb::list common_cells;
            for (auto &&c1 : v1.cells) {
                for (auto &&c2 : v2.cells) {
                    if (nb::cast<size_t>(c1) == nb::cast<size_t>(c2)) {
                        common_cells.append(c1);
                    }
                }
            }

            if (nb::len(common_cells) != 2)
                continue;

            nb::list common_vertices;
            common_vertices.append(v1);
            common_vertices.append(v2);
            auto e = Edge(common_vertices);
            e.cells = common_cells;
            e.update_mid_point();
            for (auto &&cell_id : common_cells) {
                auto cell_id_cast = nb::cast<size_t>(cell_id);
                nb::cast<Cell&>(cells[cell_id_cast-1]).edges.append(e);
                nb::cast<Cell&>(cells[cell_id_cast-1]).vertices.append(v1);
                nb::cast<Cell&>(cells[cell_id_cast-1]).vertices.append(v2);
            }
        }
    }

    for (auto &&c : cells) {
        nb::cast<Cell&>(c).update_centroid_from_vertices();
        nb::cast<Cell&>(c).update_neighbors_from_vertices();
    }

    return cells;
}

NB_MODULE(tissue, m) {
    m.doc() = "nanobind example plugin"; // optional module docstring

    m.def("rebuild_tissue", &rebuild_tissue, "...", "image"_a);

    nb::class_<Vertex>(m, "Vertex")
        .def(nb::init<double, double, double>())
        .def("__repr__", &Vertex::to_string)
        .def("__str__", &Vertex::to_string)

        .def("__lt__", &Vertex::operator<)
        .def("__gt__", &Vertex::operator>)

        .def_ro("x", &Vertex::x)
        .def_ro("y", &Vertex::y)
        .def_ro("z", &Vertex::z);

    nb::class_<Edge>(m, "Edge")
        .def(nb::init<nb::list>())
        .def("__repr__", &Edge::to_string)
        .def("__str__", &Edge::to_string)

        .def_ro("mid_point", &Edge::mid_point)
        .def_ro("vertices", &Edge::vertices);

    nb::class_<Cell>(m, "Cell")
        .def(nb::init<size_t>())
        .def("__repr__", &Cell::to_string)
        .def("__str__", &Cell::to_string)

        .def_ro("id", &Cell::id)
        .def_ro("vertices", &Cell::vertices)
        .def_ro("edges", &Cell::edges)
        .def_ro("neighbors", &Cell::neighbors)
        .def_ro("centroid", &Cell::centroid);


}

