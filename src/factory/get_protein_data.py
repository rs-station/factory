import gemmi
import requests

def get_protein_data(url: str):
    response = requests.get(url)
    if not response.ok:
        raise Exception("Failed to download the file")
    cif_data = response.text

    doc = gemmi.cif.read_string(cif_data)
    block = doc.sole_block()
    structure = gemmi.make_structure_from_block(block)
    unit_cell = structure.cell
    spacegroup = structure.spacegroup_hm
    dmin_str = block.find_value("_reflns.d_resolution_high")
    if dmin_str is None:
        raise Exception("Resolution (dmin) not found in the CIF file")
    dmin = float(dmin_str)
    return{"unit_cell": unit_cell, "spacegroup": spacegroup, "dmin": dmin_str}

if __name__ == "__main__":
    data = get_protein_data("https://files.rcsb.org/download/9B7C.cif")
    print(data["unit_cell"])
    print(data["spacegroup"])
    print(data["dmin"])
