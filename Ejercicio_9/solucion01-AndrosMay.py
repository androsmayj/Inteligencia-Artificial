import re
import yaml


def parse_atom(atom_str):
    """Separa el predicado y sus argumentos: 'Estudiante(Ana)' -> 'Estudiante', ['Ana']"""
    match = re.match(r'^(\w+)\(([^)]+)\)$', atom_str.strip())
    if not match:
        return atom_str.strip(), []
    pred = match.group(1)
    args = [a.strip() for a in match.group(2).split(',')]
    return pred, args

def format_atom(pred, args):
    if not args:
        return pred
    return f"{pred}({','.join(args)})"

def is_variable(term):
    """Las variables empiezan en minúscula ('x'), las constantes en mayúscula ('Ana')"""
    return term[0].islower()

def unify_terms(term1, term2, subst):
    if subst is None:
        return None
    elif term1 == term2:
        return subst
    elif is_variable(term1):
        return unify_var(term1, term2, subst)
    elif is_variable(term2):
        return unify_var(term2, term1, subst)
    else:
        return None

def unify_var(var, val, subst):
    if var in subst:
        return unify_terms(subst[var], val, subst)
    elif val in subst:
        return unify_terms(var, subst[val], subst)
    else:
        new_subst = dict(subst)
        new_subst[var] = val
        return new_subst

def unify_atoms(atom1, atom2, subst=None):
    if subst is None:
        subst = {}
    pred1, args1 = parse_atom(atom1) if isinstance(atom1, str) else atom1
    pred2, args2 = parse_atom(atom2) if isinstance(atom2, str) else atom2

    if pred1 != pred2 or len(args1) != len(args2):
        return None

    current_subst = dict(subst)
    for a1, a2 in zip(args1, args2):
        current_subst = unify_terms(a1, a2, current_subst)
        if current_subst is None:
            return None
    return current_subst

def apply_subst(atom_str, subst):
    pred, args = parse_atom(atom_str)
    new_args = [subst.get(a, a) for a in args]
    return format_atom(pred, new_args)



def forward_chaining(hechos_iniciales, reglas):
    hechos_conocidos = list(hechos_iniciales)
    atomos_nuevos_totales = []
    ronda = 1

    print("=== INICIANDO ENCADENAMIENTO HACIA ADELANTE ===")
    
    while True:
        print(f"\n--- Ronda {ronda} ---")
        hechos_al_inicio = list(hechos_conocidos)
        nuevos_en_esta_ronda = []

        for regla in reglas:
            r_id = regla['id']
            premisas = regla['si']
            conclusion = regla['entonces']

            def match_premisas(idx_premisa, current_subst):
                if idx_premisa == len(premisas):
                    yield current_subst
                    return

                premisa = premisas[idx_premisa]
                for hecho in hechos_al_inicio:
                    subst = unify_atoms(premisa, hecho, current_subst)
                    if subst is not None:
                        yield from match_premisas(idx_premisa + 1, subst)

            for subst in match_premisas(0, {}):
                atomo_derivado = apply_subst(conclusion, subst)
                if atomo_derivado not in hechos_conocidos and atomo_derivado not in nuevos_en_esta_ronda:
                    nuevos_en_esta_ronda.append(atomo_derivado)
                    print(f"  Fire {r_id} {subst} => {atomo_derivado}")

        if not nuevos_en_esta_ronda:
            print("  No new facts.")
            break

        for nuevo in nuevos_en_esta_ronda:
            hechos_conocidos.append(nuevo)
            atomos_nuevos_totales.append(nuevo)
        ronda += 1

    return hechos_conocidos, atomos_nuevos_totales



def backward_chaining(consulta, hechos, reglas):
    print(f"\n=== EVALUANDO CONSULTA HACIA ATRÁS: {consulta} ===")
    
    def bc_or(meta, hechos, reglas, pila_metas, nivel=0):
        indent = "  " * nivel
        print(f"{indent}goal  {meta}")

        if meta in pila_metas:
            print(f"{indent}  ! Ciclo detectado en {meta}. Cortando.")
            return False

        for hecho in hechos:
            subst = unify_atoms(meta, hecho)
            if subst is not None:
                print(f"{indent}fact  {hecho}")
                return True

        pila_actual = pila_metas + [meta]
        for regla in reglas:
            r_id = regla['id']
            head = regla['entonces']
            subst = unify_atoms(meta, head)
            
            if subst is not None:
                print(f"{indent}rule  {r_id} {subst}")
                premisas_subst = [apply_subst(p, subst) for p in regla['si']]
                
                exito_premisas = True
                for premisa in premisas_subst:
                    if not bc_or(premisa, hechos, reglas, pila_actual, nivel + 1):
                        exito_premisas = False
                        break
                
                if exito_premisas:
                    return True

        return False

    resultado = bc_or(consulta, hechos, reglas, [])
    return resultado



if __name__ == "__main__":
    with open("biblioteca.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    hechos = data["hechos"]
    reglas = data["reglas"]

    hechos_finales, atomos_nuevos = forward_chaining(hechos, reglas)

    consultas = ["AccesoNoche(Ana)", "AccesoNoche(Luis)", "AccesoNoche(Marta)"]

    print("  RESULTADOS CONSULTAS (FORWARD CHAINING):\n")
    for q in consultas:
        probada = q in hechos_finales
        print(f"Query {q}: {'PROBADA: SÍ' if probada else 'PROBADA: NO'}")

    print("  EJECUCIÓN BACKWARD CHAINING:\n")
    for q in consultas:
        probada = backward_chaining(q, hechos, reglas)
        print(f"Query proved: {q} -> {'YES' if probada else 'NO'}")