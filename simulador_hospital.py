import subprocess
from dataclasses import dataclass
from heapq import heappush, heappop, heapify
from math import inf
from time import perf_counter

PRIORIDADES = {
    "emergencia": 1000,
    "urgente": 100,
    "leve": 10
}

@dataclass
class Paciente:
    Id: int 
    Nombre: str
    Llegada: int
    Duracion: int 
    Prioridad: str
    NecesitaCama: bool = False
    NecesitaDiagnostico: bool = False
    inicio: int = -1 
    fin: int = -1 
    espera: int = 0 
    estado: str = "Pendiente"
    @property
    def puntuacion(self):
        return PRIORIDADES[self.Prioridad]

def calcular_beneficio(p, ahora):
    espera_actual = max(0, ahora - p.Llegada)
    return p.puntuacion + min(espera_actual, 60)

def optimizar_asignacion_Programacion_Dinamica(pacientes, medicos, camas, diagnosticos, ahora):
    dp = {(0, 0, 0): (0, ())}
    for i, p in enumerate(pacientes):
        nuevos = dict(dp)
        beneficio = calcular_beneficio(p, ahora)
        for (m, c, d), (valor, seleccion) in dp.items():
            nm = m + 1
            nc = c + int(p.NecesitaCama)
            nd = d + int(p.NecesitaDiagnostico)
            if nm > medicos or nc > camas or nd > diagnosticos:
                continue
            estado = (nm, nc, nd)
            candidato = (valor + beneficio, seleccion + (i,))
            if estado not in nuevos or candidato[0] > nuevos[estado][0]:
                nuevos[estado] = candidato
        dp = nuevos
    mejor = max(dp.values(), key=lambda x: x[0])
    return list(mejor[1]), mejor[0]

def optimizar_divide_y_venceras(pacientes, medicos, camas, diagnosticos, ahora):
    def generar_combinaciones(inicio, fin):
        combinaciones = []
        def explorar(i, usados_m, usadas_c, usados_d, beneficio, seleccion):
            if (usados_m > medicos or usadas_c > camas or usados_d > diagnosticos):
                return
            if i == fin:
                combinaciones.append((usados_m, usadas_c, usados_d, beneficio, seleccion))
                return
            paciente = pacientes[i]
            explorar(i + 1, usados_m, usadas_c, usados_d, beneficio, seleccion) # primero no incluir al paciente
            ## segundo, si incluir al paciente
            explorar(i + 1, usados_m + 1, usadas_c + int(paciente.NecesitaCama), usados_d + int(paciente.NecesitaDiagnostico), beneficio + calcular_beneficio(paciente, ahora), seleccion + (i,))
        explorar(inicio, 0, 0, 0, 0, ())
        return combinaciones
    mitad = len(pacientes) // 2
    izquierda = generar_combinaciones(0, mitad)
    derecha = generar_combinaciones(mitad, len(pacientes))
    mejor_beneficio = -1
    mejor_seleccion = ()
    for izq in izquierda:
        for der in derecha:
            total_m = izq[0] + der[0]
            total_c = izq[1] + der[1]
            total_d = izq[2] + der[2]
            if (total_m <= medicos and total_c <= camas and total_d <= diagnosticos):
                beneficio_total = izq[3] + der[3]
                if beneficio_total > mejor_beneficio:
                    mejor_beneficio = beneficio_total
                    mejor_seleccion = izq[4] + der[4]
    return list(mejor_seleccion), max(0, mejor_beneficio)


def comparar_algoritmos(pacientes, medicos, camas, diagnosticos, ahora):
    inicio = perf_counter()
    seleccion_pd, beneficio_pd = optimizar_asignacion_Programacion_Dinamica(pacientes, medicos, camas, diagnosticos, ahora)
    tiempo_pd = perf_counter() - inicio
    inicio = perf_counter()
    seleccion_dv, beneficio_dv = optimizar_divide_y_venceras(pacientes, medicos, camas, diagnosticos, ahora)
    tiempo_dv = perf_counter() - inicio
    def mostrar_detalle(nombre, seleccion, beneficio, duracion):
        elegidos = [pacientes[i] for i in seleccion]
        usadas_camas = sum(p.NecesitaCama for p in elegidos)
        usados_diag = sum(p.NecesitaDiagnostico for p in elegidos)
        print(f"\n{nombre}")
        print(f"Beneficio total: {beneficio}")
        print(f"Pacientes seleccionados: {len(elegidos)}")
        print(f"Médicos utilizados: {len(elegidos)}/{medicos}")
        print(f"Camas utilizadas: {usadas_camas}/{camas}")
        print(f"Equipos de diagnóstico utilizados: {usados_diag}/{diagnosticos}")
        print(f"Tiempo de ejecución: {duracion:.8f} segundos")
        print("Selección:", ", ".join(p.Nombre for p in elegidos) or "Ninguno")

    print(f"\n{"COMPARACION DE ALGORITMOS".center(50,"=")}\n")
    print(f"Pacientes candidatos: {len(pacientes)} | Momento: minuto {ahora}")
    mostrar_detalle("PROGRAMACIÓN DINÁMICA:", seleccion_pd, beneficio_pd, tiempo_pd)
    mostrar_detalle("DIVIDE Y VENCERÁS:", seleccion_dv, beneficio_dv, tiempo_dv)
    print(f"\n{"CONCLUSION DE LA COMPARACION".center(50,"=")}\n")
    if beneficio_pd == beneficio_dv:
        print("Ambos algoritmos encontraron el mismo beneficio optimo")
    else:
        print("Atencion!!!! los resultaos difieren")
    if tiempo_pd < tiempo_dv:
        print("En esta prueba la Programación Dinamica fue mas rapida.")
    elif tiempo_dv < tiempo_pd:
        print("En esta prueba Divide y Venceras fue más rapido")
    else:
        print("Ambos tardaron practicamente lo mismo")
    print("="*50)
    return seleccion_pd, seleccion_dv

###############################################################################################################################################
class Hospital:
    def __init__(self, medicos=10, camas=5, diagnosticos=3):
        self.total_medicos = medicos
        self.total_camas = camas
        self.total_diagnosticos = diagnosticos
        self.pacientes = []
        self.siguiente_id = 1
        self.tiempo = 0
        self.ejecutado = False

    def registrar_paciente(self, nombre, llegada, duracion, prioridad, necesita_cama=False, necesita_diagnostico=False):
        if llegada < 0 or duracion <= 0:
            raise ValueError("La llegada no puede ser negativa y la duración debe ser mayor que cero.")
        if necesita_cama and self.total_camas == 0:
            raise ValueError("El hospital no dispone de camas.")
        if necesita_diagnostico and self.total_diagnosticos == 0:
            raise ValueError("No hay equipos de diagnóstico disponibles.")
        p = Paciente(Id=self.siguiente_id, Nombre=nombre, Llegada=llegada, Duracion=duracion,Prioridad=prioridad, NecesitaCama=necesita_cama, NecesitaDiagnostico=necesita_diagnostico)
        self.pacientes.append(p)
        self.siguiente_id += 1
        print(f"Paciente {p.Id} registrado: {p.Nombre} | LLEGADA: {p.Llegada} min | DURACIÓN: {p.Duracion} min | PRIORIDAD: {p.Prioridad} | Cama: {p.NecesitaCama} | Diagnostico: {p.NecesitaDiagnostico}")

    def simular(self, escoger_algoritmo):
        if self.ejecutado:
            print("La simulación ya ha sido procesada. Reinicie para simular nuevamente.")
            return
        self.ejecutado = True
        llegadas = []
        for p in self.pacientes:
            heappush(llegadas, (p.Llegada, p.Id, p)) 
        espera = []
        activos = []
        self.tiempo = 0
        print(f"\n{"INICIO DE LA SIMULACION".center(50,"-")}\n")
        while llegadas or espera or activos:
            terminados = [(fin, p) for fin, p in activos if fin <= self.tiempo]
            for fin, p in terminados:
                p.fin = fin
                p.estado = "Atendido"
                print(f"[{fin:>3} min] Termina atención de {p.Nombre} (esperó {p.espera} min).")
            activos = [(fin, p) for fin, p in activos if fin > self.tiempo] 
            while llegadas and llegadas[0][0] <= self.tiempo:
                _, _, p = heappop(llegadas)
                heappush(espera, (-p.puntuacion, p.Llegada, p.Id, p))
                print(f"[{self.tiempo:>3} min] Llega a urgencias: {p.Nombre} [{p.Prioridad.upper()}]")
            medicos_libres = self.total_medicos - len(activos)
            camas_libres = self.total_camas - sum(p.NecesitaCama for _, p in activos)
            diagnosticos_libres = self.total_diagnosticos - sum(p.NecesitaDiagnostico for _, p in activos)
            espera = [(-(p.puntuacion + min(max(0, self.tiempo - p.Llegada), 60)), p.Llegada, p.Id, p) for _, _, _, p in espera]
            heapify(espera)
            candidatos = [elemento[3] for elemento in espera]

            if escoger_algoritmo == 1:
                seleccion, beneficio = optimizar_asignacion_Programacion_Dinamica(candidatos, medicos_libres, camas_libres, diagnosticos_libres, self.tiempo)
            elif escoger_algoritmo == 2:
                seleccion, beneficio = optimizar_divide_y_venceras(candidatos, medicos_libres, camas_libres, diagnosticos_libres, self.tiempo)
                 
            if seleccion:
                elegidos = {candidatos[i].Id for i in seleccion} 
                espera = [elemento for elemento in espera if elemento[3].Id not in elegidos]
                heapify(espera)
                for i in seleccion:
                    p = candidatos[i] 
                    p.inicio = self.tiempo 
                    p.fin = self.tiempo + p.Duracion 
                    p.espera = p.inicio - p.Llegada
                    p.estado = "En atención" 
                    activos.append((p.fin, p)) 
                    print(f"[{self.tiempo:>3} min] Inicia atención de {p.Nombre} hasta el min {p.fin}") 
                continue 
            proximas_llegadas = (llegadas[0][0] if llegadas else inf) 
            proximas_finalizaciones = (min(fin for fin, _ in activos) if activos else inf)
            siguiente_evento = min(proximas_llegadas, proximas_finalizaciones)
            if siguiente_evento == inf:
                break
            if siguiente_evento <= self.tiempo:
                raise RuntimeError("No se pudo avanzar el tiempo de simulación.")
            self.tiempo = siguiente_evento
        print(f"\n{"FIN DE LA SIMULACION".center(50,"-")}\n")
        self.mostrar_reporte()

    def reiniciar(self):
        self.tiempo = 0
        self.ejecutado = False
        for p in self.pacientes:
            p.inicio = -1
            p.fin = -1
            p.espera = 0
            p.estado = "Pendiente"
        print("El sistema del hospital ha sido reiniciado.")

    def mostrar_reporte(self):
        atendidos = [p for p in self.pacientes if p.estado == "Atendido"]
        pendientes = [p for p in self.pacientes if p.estado != "Atendido"]
        print("REPORTE ESTADÍSTICO:")
        print(f"Pacientes registrados: {len(self.pacientes)}")
        print(f"Pacientes atendidos:   {len(atendidos)}")
        print(f"Pacientes pendientes:  {len(pendientes)}")
        if atendidos:
            esperas = [p.espera for p in atendidos]
            print(f"Espera promedio:       {sum(esperas) / len(esperas):.2f} min")
            print(f"Espera mínima:         {min(esperas)} min")
            print(f"Espera máxima:         {max(esperas)} min")
        print("\nDetalle por paciente:")
        for p in sorted(self.pacientes, key=lambda x: x.Id):
            print(f"#{p.Id} {p.Nombre:15} | {p.Prioridad:10} | Estado: {p.estado:12} | Espera: {p.espera if p.inicio >= 0 else '-'}")

###############################################################################################################################################
def datos_demostracion(hospital):
    datos = [
        ("Paul", 0, 30, "emergencia", True, False), ("Efrain", 0, 20, "leve", False, False), ("Cristopher", 2, 25, "urgente", False, True),
        ("Jhonny Pacheco", 3, 15, "emergencia", False, True), ("Carlos Alcantara", 4, 35, "leve", True, False), ("Nicola", 5, 10, "urgente", False, False),
        ("Mateo", 0, 25, "emergencia", True, False), ("Lucia", 0, 30, "urgente", False, True), ("Rodrigo", 0, 15, "leve", False, False),
        ("Valeria", 0, 20, "emergencia", True, True), ("Diego", 0, 40, "urgente", True, False), ("Sofia", 0, 10, "leve", False, False),
        ("Joaquin", 0, 25, "emergencia", False, False), ("Camila", 0, 20, "urgente", False, False), ("Alejandro", 0, 35, "leve", True, False),
        ("Valentina", 0, 15, "urgente", False, True), ("Renata", 0, 25, "emergencia", True, False), ("Mauricio", 0, 30, "leve", False, False), 
        ("Fernando", 2, 20, "urgente", True, True), ("Mariana", 3, 15, "emergencia", False, True), ("Gabriel", 5, 10, "leve", False, False),
        ("Lolo Fernandez", 12, 24, "urgente", True, True), ("Michael J", 10, 25, "urgente", True, True), ("Bruno M", 20, 36, "leve", False, False)
    ]
    for d in datos:
        hospital.registrar_paciente(*d)

def leer_numero(mensaje):
    while True:
        try:
            valor = int(input(mensaje))
            if valor < 0:
                print(f"El valor mínimo es 0")
                continue
            return valor
        except ValueError:
            print("Ingresa un número entero válido.")

def leer_booleano(mensaje):
    while True:
        respuesta = input(mensaje).strip().lower()
        if respuesta == "s":
            return True
        elif respuesta == "n":
            return False
        else:
            print("Ingresa 'S' para Sí o 'N' para No.")

def leer_prioridad():
    while True:
        entrada = input("Prioridad: [1] Emergencia | [2] Urgente | [3] Leve: ")
        if entrada == "1":
            return "emergencia"
        elif entrada == "2":
            return "urgente"
        elif entrada == "3":
            return "leve"
        else:
            print("Ingrese un número del 1 al 3.")

def menu():
    hospital = Hospital(medicos=10, camas=5, diagnosticos=3)
    while True:
        print(f"\n{"SISTENA DE COLA DE ATENCION HOSPITALARIA".center(80,"*")}")
        print(f"| Médicos: {hospital.total_medicos} | Camas: {hospital.total_camas} | Diagnósticos: {hospital.total_diagnosticos}\n|")
        print("| [1] Registrar paciente\n| [2] Ejecutar simulacion y comparar algoritmos\n| [3] Reiniciar simulación\n| [0] Salir\n|")
        print("| Presione [D] para cargar datos de demostración y ejecutar")
        print("*"*70)
        opcion = input("Elige una opción: ").strip()
        
        if opcion == "1":
            if hospital.ejecutado:
                print("La simulación ya fue procesada, reinicie el programa.")
                input()
                continue
            nombre = input("Nombre del paciente: ").strip()
            if not nombre:
                print("El nombre no puede estar vacío.")
                input()
                continue
            llegada = leer_numero("Minuto de llegada: ")
            duracion = leer_numero("Duración de atención (en minutos): ")
            prioridad = leer_prioridad()
            necesitaCama = leer_booleano("¿Necesita cama? (S/N): ")
            necesitaDiagnostico = leer_booleano("¿Necesita diagnóstico? (S/N): ")
            try:
                hospital.registrar_paciente(nombre, llegada, duracion, prioridad, necesitaCama, necesitaDiagnostico)
            except ValueError as error:
                print("Hay error:", error)

        elif opcion == "2":
            if not hospital.pacientes:
                print("La lista de pacientes está vacía.")
            else:
                candidatos_iniciales = [p for p in hospital.pacientes if p.Llegada <= hospital.tiempo]
                comparar_algoritmos(candidatos_iniciales, hospital.total_medicos, hospital.total_camas, hospital.total_diagnosticos, hospital.tiempo)
                print(f"\n{"CON PROGRAMACION DINAMICA".center(80, "/")}\n")
                hospital.simular(1)
                hospital.reiniciar()
                print(f"\n{"CON DIVIDE Y VENCERAS".center(80, "/")}\n")
                hospital.simular(2)
                
        elif opcion == "3":
            hospital.reiniciar()

        elif opcion.lower() == "d":
            hospitalSimulacion = Hospital(medicos=10, camas=5, diagnosticos=3)
            print(f"\n{"DEMOSTRACION INICIADA".center(80,"*")}\n")
            datos_demostracion(hospitalSimulacion)
            comparar_algoritmos(hospitalSimulacion.pacientes, hospitalSimulacion.total_medicos, hospitalSimulacion.total_camas, hospitalSimulacion.total_diagnosticos, ahora=0)
            print(f"\n{"CON PROGRAMACION DINAMICA".center(80, "/")}\n")
            hospitalSimulacion.simular(1)
            hospitalSimulacion.reiniciar()
            print(f"\n{"CON DIVIDE Y VENCERAS".center(80, "/")}\n")
            hospitalSimulacion.simular(2)
            print(f"\n{"DEMOSTRACION FINALIZADA".center(80,"*")}\n")

        elif opcion == "0":
            print("Saliendo...")
            input()
            subprocess.run("cls", shell=True)
            break
        else:
            print("Opción inválida.")

        input("\nPresione cualquie tecla para continuar: ")
        subprocess.run("cls", shell=True)

if __name__ == "__main__":
    menu()

#Comentario para probar