import uuid

from datapizza.type import Chunk

from src.parsing import MenuData, load_dish_id_mapping


def _dish_chunk_text(dish, menu: MenuData) -> str:
    """Build a natural-language chunk for a single dish.

    Includes all context needed for retrieval: restaurant, planet, chef,
    licenses, LTK, ingredients, and techniques.
    """
    ingredients = ", ".join(dish.ingredients) if dish.ingredients else "none listed"
    techniques = ", ".join(dish.techniques) if dish.techniques else "none listed"
    licenses_str = ", ".join(f"{lic.type} {lic.grade}" for lic in menu.licenses)

    return (
        f"Dish: {dish.name}\n"
        f"Restaurant: {menu.restaurant}\n"
        f"Planet: {menu.planet}\n"
        f"Chef: {menu.chef}\n"
        f"Chef licenses: {licenses_str}\n"
        f"Restaurant LTK level: {menu.ltk}\n"
        f"Ingredients: {ingredients}\n"
        f"Techniques: {techniques}"
    )


def menus_to_chunks(menus: list[MenuData]) -> list[Chunk]:
    """Convert parsed menus into one Chunk per dish.

    Each chunk contains:
    - text: natural-language description with full context
    - metadata: structured fields (dish_id, restaurant, planet, chef,
      ingredients, techniques, licenses, ltk, source_file)
    - id: deterministic UUID derived from dish name
    """
    dish_ids = load_dish_id_mapping()
    chunks: list[Chunk] = []

    for menu in menus:
        for dish in menu.dishes:
            dish_id = dish_ids.get(dish.name)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, dish.name))

            chunk = Chunk(
                id=chunk_id,
                text=_dish_chunk_text(dish, menu),
                metadata={
                    "dish_id": dish_id,
                    "dish_name": dish.name,
                    "restaurant": menu.restaurant,
                    "planet": menu.planet,
                    "chef": menu.chef,
                    "licenses": [
                        {"type": lic.type, "grade": lic.grade}
                        for lic in menu.licenses
                    ],
                    "ltk": menu.ltk,
                    "ingredients": dish.ingredients,
                    "techniques": dish.techniques,
                    "source_file": menu.source_file,
                },
            )
            chunks.append(chunk)

    return chunks
