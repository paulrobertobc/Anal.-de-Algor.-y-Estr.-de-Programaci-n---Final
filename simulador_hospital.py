import subprocess
from dataclasses import dataclass, field
from heapq import heappush, heappop, heapify
from time import perf_counter
from typing import Optional

PRIORIDADES = {"emergencia": 1000, "urgente": 100, "leve": 10}
ESPECIALIDADES = ("cardiologia", "pediatria", "traumatologia", "medicina general")

def hora_a_minutos(hora: str) -> int: #recibe un string y devuele un entero
    try:
        h, m = map(int, hora.split(":")) #con split vamos a cortar ese string en dos partes. despues con el map le decimos que cada parte sera un int
    except (ValueError, AttributeError):
        raise ValueError("La hora debe tener formato HH:MM, por ejemplo 08:00.")
    if not (0 <= h <= 23 and 0 <= m <= 59): #evaluamos si h entra dentro del rango 24 (porquw son las horas) y m entra en el rango de 59 (porque son los minutos)
        raise ValueError("La hora debe estar entre 00:00 y 23:59.")
    return h * 60 + m #al final devuelve el calculo con los minutos finales

def minutos_a_hora(minutos: int) -> str: #recibe un int y devuelve un string
    minutos = minutos % (24 * 60) # dividimos los minutos entre 24 * 60 (que es 1440, los minutos totales de un dia) para mantenernos en el rango de un dia
    return f"{minutos // 60:02d}:{minutos % 60:02d}" #devolvemos las horas con la division entera. delvolvemos los minutos restantes usando el modulo (%)


@dataclass
class Turno:
    hora_inicio: str
    hora_fin: str

    def __post_init__(self):
        inicio = hora_a_minutos(self.hora_inicio)
        fin = hora_a_minutos(self.hora_fin)
        if fin == inicio:
            raise ValueError("El inicio y el fin del turno no pueden ser iguales.")

    @property
    def inicio(self):
        return hora_a_minutos(self.hora_inicio)

    @property
    def fin(self):
        return hora_a_minutos(self.hora_fin)

    @property
    def fin_relativo(self):
        """Fin del turno en la misma jornada; suma 24 h si cruza medianoche."""
        fin = self.fin
        return fin + 24 * 60 if fin < self.inicio else fin

@dataclass
class Medico:
    Id: int
    Nombre: str
    Especialidad: str
    Turno: Turno
    estado: str = "Fuera de turno"
    consultorio_id: Optional[int] = None
    ocupado_hasta: int = -1
    extension: bool = False
    fin_turno_asignado: int = -1

@dataclass
class Consultorio:
    Id: int
    ocupado_por: Optional[int] = None

@dataclass
class Paciente:
    Id: int
    Nombre: str
    Llegada: int  # minutos desde el inicio de la jornada simulada
    Duracion: int
    Prioridad: str
    Especialidad: str = "medicina general"
    NecesitaCama: bool = False
    NecesitaDiagnostico: bool = False
    inicio: int = -1
    fin: int = -1
    espera: int = 0
    estado: str = "Pendiente"
    medico_id: Optional[int] = None
    consultorio_id: Optional[int] = None

    @property
    def puntuacion(self):
        return PRIORIDADES[self.Prioridad]

def calcular_beneficio(p, ahora):
    espera_actual = max(0, ahora - p.Llegada)
    return p.puntuacion + min(espera_actual, 60)

def optimizar_programacion_dinamica(pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora):
    """DP exacta: cada estado reserva médicos concretos mediante una máscara de bits."""
    # (máscara médicos, camas, diagnósticos) -> (beneficio, tupla (índice paciente, índice médico))
    dp = {(0, 0, 0): (0, ())}
    for i, p in enumerate(pacientes):
        nuevos = dict(dp)  # alternativa: no seleccionar al paciente
        beneficio = calcular_beneficio(p, ahora)
        for (mascara, usadas_camas, usados_diag), (valor, asignaciones) in dp.items():
            if mascara.bit_count() >= consultorios:
                continue
            for indice_medico in medicos_elegibles.get(i, []):
                bit = 1 << indice_medico
                if mascara & bit:
                    continue
                nc = usadas_camas + int(p.NecesitaCama)
                nd = usados_diag + int(p.NecesitaDiagnostico)
                if nc > camas or nd > diagnosticos:
                    continue
                estado = (mascara | bit, nc, nd)
                candidato = (valor + beneficio, asignaciones + ((i, indice_medico),))
                if estado not in nuevos or candidato[0] > nuevos[estado][0]:
                    nuevos[estado] = candidato
        dp = nuevos
    mejor = max(dp.values(), key=lambda x: x[0])
    return list(mejor[1]), mejor[0]


def optimizar_divide_y_venceras(pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora):
    """Divide pacientes en dos mitades, genera fronteras de estados y combina máscaras compatibles."""
    def generar_estados(inicio, fin):
        estados = {(0, 0, 0): (0, ())}
        for i in range(inicio, fin):
            p = pacientes[i]
            nuevos = dict(estados)
            beneficio = calcular_beneficio(p, ahora)
            for (mascara, usadas_camas, usados_diag), (valor, asignaciones) in estados.items():
                if mascara.bit_count() >= consultorios:
                    continue
                for indice_medico in medicos_elegibles.get(i, []):
                    bit = 1 << indice_medico
                    if mascara & bit:
                        continue
                    nc = usadas_camas + int(p.NecesitaCama)
                    nd = usados_diag + int(p.NecesitaDiagnostico)
                    if nc > camas or nd > diagnosticos:
                        continue
                    estado = (mascara | bit, nc, nd)
                    candidato = (valor + beneficio, asignaciones + ((i, indice_medico),))
                    if estado not in nuevos or candidato[0] > nuevos[estado][0]:
                        nuevos[estado] = candidato
            estados = nuevos
        return estados

    mitad = len(pacientes) // 2
    izquierda = generar_estados(0, mitad)
    derecha = generar_estados(mitad, len(pacientes))
    mejor_beneficio = 0
    mejor_asignacion = ()
    for (mascara_i, camas_i, diag_i), (valor_i, asign_i) in izquierda.items():
        for (mascara_d, camas_d, diag_d), (valor_d, asign_d) in derecha.items():
            if mascara_i & mascara_d:
                continue
            if (mascara_i | mascara_d).bit_count() > consultorios:
                continue
            if camas_i + camas_d > camas or diag_i + diag_d > diagnosticos:
                continue
            total = valor_i + valor_d
            if total > mejor_beneficio:
                mejor_beneficio = total
                mejor_asignacion = asign_i + asign_d
    return list(mejor_asignacion), mejor_beneficio


def comparar_algoritmos(pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora):
    inicio = perf_counter()
    asign_pd, beneficio_pd = optimizar_programacion_dinamica(
        pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora)
    tiempo_pd = perf_counter() - inicio
    inicio = perf_counter()
    asign_dv, beneficio_dv = optimizar_divide_y_venceras(
        pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora)
    tiempo_dv = perf_counter() - inicio

    def detalle(nombre, asignaciones, beneficio, duracion):
        elegidos = [pacientes[i] for i, _ in asignaciones]
        print(f"\n            {nombre}            ")
        print(f"- Beneficio total: {beneficio}")
        print(f"- Pacientes seleccionados: {len(elegidos)}")
        print(f"- Médicos utilizados: {len(asignaciones)}")
        print(f"- Camas utilizadas: {sum(p.NecesitaCama for p in elegidos)}/{camas}")
        print(f"- Equipos de diagnóstico utilizados: {sum(p.NecesitaDiagnostico for p in elegidos)}/{diagnosticos}")
        print(f"- Consultorios utilizados: {len(asignaciones)}/{consultorios}")
        print(f"- Tiempo de ejecución: {duracion:.8f} segundos")
        print("Selección:", ", ".join(p.Nombre for p in elegidos) or "Ninguno")

    print("\n" + "COMPARACIÓN DE ALGORITMOS".center(100, "=") + "\n")
    print(f"Pacientes candidatos: {len(pacientes)} | Minuto de simulación: {ahora}")
    detalle("PROGRAMACIÓN DINÁMICA", asign_pd, beneficio_pd, tiempo_pd)
    detalle("DIVIDE Y VENCERÁS", asign_dv, beneficio_dv, tiempo_dv)
    print("\n" + "CONCLUSIÓN DE LA COMPARACIÓN".center(100, "="))
    if beneficio_pd == beneficio_dv:
        print("| Ambos algoritmos encontraron el mismo beneficio óptimo.")
    else:
        print("| Los resultados difieren; revisa las selecciones y restricciones.")
    if tiempo_pd < tiempo_dv:
        print("| En esta prueba, Programación Dinámica fue más rápida.")
    elif tiempo_dv < tiempo_pd:
        print("| En esta prueba, Divide y Vencerás fue más rápido.")
    else:
        print("Ambos tardaron prácticamente lo mismo.")
    print("=" * 100)
    return asign_pd, asign_dv


class Hospital:
    def __init__(self, medicos=0, camas=5, diagnosticos=3, cantidad_consultorios=4):
        self.total_camas = camas
        self.total_diagnosticos = diagnosticos
        self.consultorios = [Consultorio(i + 1) for i in range(cantidad_consultorios)]
        self.medicos = []
        self.pacientes = []
        self.siguiente_id = 1
        self.siguiente_medico_id = 1
        self.tiempo = 0
        self.inicio_jornada = None  # se fija con el primer médico registrado.
        self.ejecutado = False
        datosMedicos = [
            ("Ramírez", "cardiologia", "08:00", "14:00"),#turno mañana
            ("Torres", "pediatria", "08:00", "16:00"),
            ("Vargas", "medicina general", "08:00", "14:00"),
            ("García", "traumatologia", "06:00", "14:00"),
            ("Mendoza", "traumatologia", "14:00", "20:00"), #turno tarde
            ("Castillo", "cardiologia", "14:00", "20:00"),
            ("Salas", "medicina general", "14:00", "20:00"),
            ("Vega", "pediatria", "14:00", "22:00"),
            ("Paredes", "cardiologia", "22:00", "06:00"),#turno noche
            ("León", "pediatria", "22:00", "06:00"),
            ("Rojas", "traumatologia", "22:00", "06:00"),
            ("Huamán", "medicina general", "22:00", "06:00")
        ]
        for m in datosMedicos:
            self.registrar_medico(*m)

    @property
    def total_medicos(self):
        return len(self.medicos)

    def registrar_medico(self, nombre, especialidad, hora_inicio, hora_fin):
        especialidad = especialidad.strip().lower()
        if especialidad not in ESPECIALIDADES:
            raise ValueError("Especialidad no válida.")
        turno = Turno(hora_inicio, hora_fin)
        medico = Medico(self.siguiente_medico_id, nombre.strip(), especialidad, turno)
        self.medicos.append(medico)
        self.siguiente_medico_id += 1
        if self.inicio_jornada is None:
            self.inicio_jornada = turno.inicio
        else:
            self.inicio_jornada = min(self.inicio_jornada, turno.inicio)
        return medico

    def registrar_paciente(self, nombre, llegada, duracion, prioridad, especialidad="medicina general",necesita_cama=False, necesita_diagnostico=False):
        if llegada < 0 or duracion <= 0:
            raise ValueError("La llegada no puede ser negativa y la duración debe ser mayor que cero.")
        if prioridad not in PRIORIDADES:
            raise ValueError("Prioridad no válida.")
        especialidad = especialidad.strip().lower()
        if especialidad not in ESPECIALIDADES:
            raise ValueError("Especialidad no válida.")
        if necesita_cama and self.total_camas == 0:
            raise ValueError("El hospital no dispone de camas.")
        if necesita_diagnostico and self.total_diagnosticos == 0:
            raise ValueError("No hay equipos de diagnóstico disponibles.") 
        p = Paciente(self.siguiente_id, nombre.strip(), llegada, duracion, prioridad,especialidad, necesita_cama, necesita_diagnostico)
        self.pacientes.append(p)
        self.siguiente_id += 1
        print(f"Paciente {p.Id} registrado: {p.Nombre} | "f"llegada: {self.hora_actual(p.Llegada)} | "f"{p.Prioridad} | {p.Especialidad}")
        return p

    def hora_actual(self, tiempo=None):
        if tiempo is None:
            tiempo = self.tiempo
        if self.inicio_jornada is None:
            return "00:00"
        return minutos_a_hora(self.inicio_jornada + tiempo)

    def convertir_hora_llegada(self, hora: str) -> int:
        """Convierte HH:MM a minutos relativos al inicio de la simulación."""
        if self.inicio_jornada is None:
            raise ValueError("Primero debes registrar al menos un médico.")
        minutos_reloj = hora_a_minutos(hora)
        llegada = minutos_reloj - self.inicio_jornada
        if llegada < 0:
            llegada += 24 * 60
        return llegada

    def intervalo_turno_actual(self, medico, tiempo):
        #nos devuelve un tupla (inicio, fin) del turno que cubre el minuto simulado actual.
        #los valores devueltos son minutos relativos al inicio de la simulación, también contempla turnos que comenzaron el día anterior y terminan hoy
        absoluto = self.inicio_jornada + tiempo
        dia = absoluto // (24 * 60)
        inicio_dia = medico.Turno.inicio
        fin_dia = medico.Turno.fin_relativo
        for d in (dia - 1, dia, dia + 1):
            inicio_abs = d * 24 * 60 + inicio_dia
            fin_abs = d * 24 * 60 + fin_dia
            if inicio_abs <= absoluto < fin_abs:
                return (inicio_abs - self.inicio_jornada, fin_abs - self.inicio_jornada)
        return None

    def proximo_inicio_turno(self, medico, tiempo): #proximo inicio diario del turno, expresado en tiempo de simulacion
        absoluto = self.inicio_jornada + tiempo
        dia = absoluto // (24 * 60)
        inicio_abs = dia * 24 * 60 + medico.Turno.inicio
        if inicio_abs <= absoluto:
            inicio_abs += 24 * 60
        return inicio_abs - self.inicio_jornada

    def medicos_activos(self, especialidad=None):
        return [m for m in self.medicos if especialidad is None or m.Especialidad == especialidad]

    def actualizar_estados_medicos(self):
        for m in self.medicos:
            if m.estado == "Atendiendo":
                continue
            m.estado = "Disponible" if self.intervalo_turno_actual(m, self.tiempo) else "Fuera de turno"

    def consultorio_libre(self):
        return next((c for c in self.consultorios if c.ocupado_por is None), None)

    def medicos_elegibles(self, candidatos): #devuelve los indices de médicos que pueden iniciar cada paciente en este instante
        disponibles = []
        intervalos_fin = {}
        for m in self.medicos:
            intervalo = self.intervalo_turno_actual(m, self.tiempo)
            if m.estado == "Atendiendo" or intervalo is None:
                continue
            disponibles.append(m)
            intervalos_fin[m.Id] = intervalo[1]
        mapa = {}
        for i, p in enumerate(candidatos):
            opciones = []
            for indice, m in enumerate(disponibles):
                if m.Especialidad != p.Especialidad:
                    continue
                # solo una emergencia puede terminar despurs de la hora de salida
                if self.tiempo + p.Duracion <= intervalos_fin[m.Id] or p.Prioridad == "emergencia":
                    opciones.append(indice)
            mapa[i] = opciones
        return disponibles, mapa

    def recursos_libres(self, activos):
        camas_ocupadas = sum(p.NecesitaCama for _, p, _, _ in activos)
        diag_ocupados = sum(p.NecesitaDiagnostico for _, p, _, _ in activos)
        consultorios_libres = sum(c.ocupado_por is None for c in self.consultorios)
        return self.total_camas - camas_ocupadas, self.total_diagnosticos - diag_ocupados, consultorios_libres

    def puede_llegar_medico_especialidad(self, especialidad, ahora, activos): #comprueba si hay un medico que ya atiende o cuyo turno aun no termina a iniciar
        for m in self.medicos:
            if m.Especialidad != especialidad:
                continue
            if m.estado == "Atendiendo":
                return True
            if self.intervalo_turno_actual(m, ahora) is not None:
                return True
            # si su siguiente turno comienza pronto el paciente puede esperar
            if self.proximo_inicio_turno(m, ahora) - ahora <= 60:
                return True
        return False

    def trasladar_sin_cobertura(self, espera, activos):
        restantes = []
        for p in espera:
            if not self.puede_llegar_medico_especialidad(p.Especialidad, self.tiempo, activos):
                p.estado = "Trasladado"
                p.fin = self.tiempo
                print(f"[{self.hora_actual()}] TRASLADO: {p.Nombre} requiere {p.Especialidad}; no hay especialista disponible. Se deriva a otro hospital.")
            else:
                restantes.append(p)
        return restantes

    def asignar_consultorio(self, medico, paciente):
        consultorio = self.consultorio_libre()
        if consultorio is None:
            return False
        consultorio.ocupado_por = medico.Id
        medico.consultorio_id = consultorio.Id
        medico.estado = "Atendiendo"
        paciente.medico_id = medico.Id
        paciente.consultorio_id = consultorio.Id
        return True

    def simular(self, escoger_algoritmo):
        if self.ejecutado:
            print("La simulación ya fue procesada. Reinicie antes de volver a ejecutarla.")
            return
        if not self.medicos:
            print("Registra al menos un médico antes de iniciar la simulación.")
            return
        self.ejecutado = True
        llegadas = []
        for p in self.pacientes:
            heappush(llegadas, (p.Llegada, p.Id, p))
        espera = []
        activos = []  # (fin, paciente, médico, consultorio)
        self.tiempo = 0
        print("\n" + "INICIO DE LA SIMULACIÓN".center(100, "-") + "\n")

        while llegadas or espera or activos:
            # primero finalizar atenciones cuyo tiempo ya termino
            terminados = [a for a in activos if a[0] <= self.tiempo]
            for fin, p, medico, consultorio in terminados:
                p.fin = fin
                p.estado = "Atendido"
                medico.estado = "Disponible" if self.intervalo_turno_actual(medico, self.tiempo) else "Fuera de turno"
                medico.consultorio_id = None
                medico.ocupado_hasta = -1
                extra = " (turno extendido por emergencia)" if medico.extension else ""
                medico.extension = False
                medico.fin_turno_asignado = -1
                consultorio.ocupado_por = None
                print(f"[{self.hora_actual(fin)}] Finaliza {p.Nombre} con Dr./Dra. {medico.Nombre}, consultorio {consultorio.Id}{extra}.")
            activos = [a for a in activos if a[0] > self.tiempo]

            # segundo, registrar todas las llegadas del instante
            while llegadas and llegadas[0][0] <= self.tiempo:
                _, _, p = heappop(llegadas)
                p.estado = "En espera"
                heappush(espera, (-p.puntuacion, p.Llegada, p.Id, p))
                print(f"[{self.hora_actual()}] Llega {p.Nombre}: {p.Prioridad.upper()} | requiere {p.Especialidad}.")

            self.actualizar_estados_medicos()
            espera = [(-p.puntuacion, p.Llegada, p.Id, p) for _, _, _, p in espera]
            heapify(espera)
            candidatos = [item[3] for item in espera]
            disponibles, mapa_elegibles = self.medicos_elegibles(candidatos)
            camas_libres, diagnosticos_libres, consultorios_libres = self.recursos_libres(activos)

            if candidatos and disponibles and consultorios_libres > 0:
                if escoger_algoritmo == 1:
                    asignaciones, _ = optimizar_programacion_dinamica(
                        candidatos, mapa_elegibles, camas_libres, diagnosticos_libres,
                        consultorios_libres, self.tiempo)
                else:
                    asignaciones, _ = optimizar_divide_y_venceras(
                        candidatos, mapa_elegibles, camas_libres, diagnosticos_libres,
                        consultorios_libres, self.tiempo)

                if asignaciones:
                    elegidos = {candidatos[i].Id for i, _ in asignaciones}
                    espera = [item for item in espera if item[3].Id not in elegidos]
                    heapify(espera)
                    for indice_paciente, indice_medico in asignaciones:
                        p = candidatos[indice_paciente]
                        medico = disponibles[indice_medico]
                        consultorio = self.consultorio_libre()
                        if consultorio is None:
                            # protección adicional, normalmente no ocurre por la restriccion del algoritmo
                            heappush(espera, (-p.puntuacion, p.Llegada, p.Id, p))
                            continue
                        self.asignar_consultorio(medico, p)
                        p.inicio = self.tiempo
                        p.fin = self.tiempo + p.Duracion
                        p.espera = p.inicio - p.Llegada
                        p.estado = "En atención"
                        medico.ocupado_hasta = p.fin
                        intervalo = self.intervalo_turno_actual(medico, self.tiempo)
                        fin_turno = intervalo[1] if intervalo else self.tiempo
                        medico.fin_turno_asignado = fin_turno
                        medico.extension = p.fin > fin_turno
                        if medico.extension:
                            print(f"[{self.hora_actual()}] AVISO: Dr./Dra. {medico.Nombre} extenderá su turno para terminar esta emergencia.")
                        activos.append((p.fin, p, medico, consultorio))
                        print(f"[{self.hora_actual()}] Inicia {p.Nombre} con Dr. (a) {medico.Nombre} ({p.Especialidad.title()}) en consultorio {consultorio.Id}; termina {self.hora_actual(p.fin)}.")
                    continue

            espera_pacientes = [item[3] for item in espera]
            trasladados_o_espera = self.trasladar_sin_cobertura(espera_pacientes, activos)
            ids_restantes = {p.Id for p in trasladados_o_espera}
            espera = [item for item in espera if item[3].Id in ids_restantes]
            heapify(espera)
            # tercero, avanzar al proximo evento, que es llegada, fin de atención o inicio de turno
            proximos = []
            if llegadas:
                proximos.append(llegadas[0][0])
            if activos:
                proximos.append(min(a[0] for a in activos))
            for m in self.medicos:
                if m.estado == "Atendiendo":
                    continue
                inicio_siguiente = self.proximo_inicio_turno(m, self.tiempo)
                if inicio_siguiente > self.tiempo:
                    proximos.append(inicio_siguiente)
                intervalo = self.intervalo_turno_actual(m, self.tiempo)
                if intervalo and intervalo[1] > self.tiempo:
                    proximos.append(intervalo[1])
            proximos = [t for t in proximos if t > self.tiempo]
            if not proximos:
                # no quedan eventos que puedan liberar recursos o incorporar medicos.
                for _, _, _, p in espera:
                    p.estado = "Sin atención"
                    print(f"[{self.hora_actual()}] No fue posible atender a {p.Nombre}; queda sin atención en esta simulación.")
                espera.clear()
                break
            self.tiempo = min(proximos)

        print("\n" + "FIN DE LA SIMULACIÓN".center(100, "-") + "\n")
        self.mostrar_reporte()

    def reiniciar(self):
        self.tiempo = 0
        self.ejecutado = False
        for p in self.pacientes:
            p.inicio = -1
            p.fin = -1
            p.espera = 0
            p.estado = "Pendiente"
            p.medico_id = None
            p.consultorio_id = None
        for m in self.medicos:
            m.estado = "Fuera de turno"
            m.consultorio_id = None
            m.ocupado_hasta = -1
            m.extension = False
            m.fin_turno_asignado = -1
        for c in self.consultorios:
            c.ocupado_por = None
        print("El sistema hospitalario ha sido reiniciado.")

    def mostrar_medicos(self):
        print("\nPLANTILLA MÉDICA")
        for m in self.medicos:
            etiqueta = " (cruza medianoche)" if m.Turno.fin < m.Turno.inicio else ""
            print(f"#{m.Id} Dr./Dra. {m.Nombre:18} | {m.Especialidad.title():20} | {m.Turno.hora_inicio}-{m.Turno.hora_fin}{etiqueta}")

    def mostrar_reporte(self):
        atendidos = [p for p in self.pacientes if p.estado == "Atendido"]
        trasladados = [p for p in self.pacientes if p.estado == "Trasladado"]
        pendientes = [p for p in self.pacientes if p.estado not in ("Atendido", "Trasladado")]
        print("REPORTE ESTADÍSTICO: ")
        print(f"- Pacientes registrados: {len(self.pacientes)}")
        print(f"- Pacientes atendidos:   {len(atendidos)}")
        print(f"- Pacientes trasladados: {len(trasladados)}")
        print(f"- Pacientes pendientes:  {len(pendientes)}")
        if atendidos:
            esperas = [p.espera for p in atendidos]
            print(f"- Espera promedio:       {sum(esperas) / len(esperas):.2f} min")
            print(f"- Espera mínima:         {min(esperas)} min")
            print(f"- Espera máxima:         {max(esperas)} min")
        print("\nDETALLE POR PACIENTE: ")
        for p in sorted(self.pacientes, key=lambda x: x.Id):
            medico = next((m.Nombre for m in self.medicos if m.Id == p.medico_id), "-")
            consultorio = p.consultorio_id if p.consultorio_id is not None else "-"
            print(f"#{p.Id} {p.Nombre:18} | {p.Especialidad:20} | {p.estado:12} | Médico: {medico:15} | Consultorio: {consultorio} | Espera: {p.espera if p.inicio >= 0 else '-'}")


def datos_demostracion(hospital): #pacientes ejemplo para dar una demostracion
    datos = [ 
        #los numeros enteros que representan la llegada son los minutos desde que empezo la jornada
        #la jornada comienza a las 6,
        #0	 =>  06:00 a. m.
        #30	 => 06:30 a. m.
        #60	 => 07:00 a. m.
        #120	 => 08:00 a. m.
        #480	 => 02:00 p. m.
        #960	 => 10:00 p. m.
        #1080 => 12:00 a. m. (medianoche)
        #formato de objeto tipo Paciente: nombre, llegada, duracion, prioridad, necesitaCama, necesitaDiagnostico
        ("Paul", 5, 45, "emergencia", "cardiologia",True, False),
        ("Efrain", 15, 90, "leve", "pediatria", False, False), 
        ("Cristopher", 35, 35, "urgente", "traumatologia", False, True),
        ("Jhonny Pacheco", 60, 120, "emergencia", "medicina general",False, True), 
        ("Carlos Alcantara", 95, 60, "leve","cardiologia", True, False), 
        ("Nicola", 130, 40, "urgente", "pediatria",False, False),
        ("Mateo", 180, 150, "emergencia","cardiologia", True, False),
        ("Lucia", 225, 35, "urgente", "medicina general",False, True), 
        ("Rodrigo",270 , 75, "leve", "traumatologia",False, False),
        ("Valeria", 315, 50, "emergencia","pediatria", True, True), 
        ("Diego", 360, 180, "urgente", "cardiologia", True, False), 
        ("Sofia", 420, 45, "leve", "medicina general",False,False),
        ("Joaquin", 480, 90, "emergencia", "traumatologia",False, False),
        ("Camila", 540, 60, "urgente", "cardiologia",False, False), 
        ("Alejandro", 600, 120, "leve", "traumatologia",True, False),
        ("Valentina", 660, 30, "urgente", "medicina general",False, True),
        ("Renata", 720, 100, "emergencia", "pediatria",True, False),
        ("Mauricio", 780, 45, "leve", "cardiologia",False, False), 
        ("Fernando", 840, 150, "urgente", "traumatologia",True, True),
        ("Mariana", 900, 60, "emergencia", "medicina general",False, True),
        ("Gabriel", 960, 90, "leve", "cardiologia",False, False),
        ("Lolo Fernandez", 1020, 40, "urgente", "traumatologia",True, True),
        ("Michael J", 1080, 120, "urgente", "pediatria",True, True),
        ("Bruno M", 1200, 75, "leve", "medicina general",False, False),
    ]
    for d in datos:
        hospital.registrar_paciente(*d)

def leer_numero(mensaje, minimo=0):
    while True:
        try:
            valor = int(input(mensaje))
            if valor < minimo:
                print(f"El valor mínimo es {minimo}.")
                continue
            return valor
        except ValueError:
            print("Ingresa un número entero válido")

def leer_booleano(mensaje):
    while True:
        respuesta = input(mensaje).strip().lower()
        if respuesta in ("s", "si", "sí"):
            return True
        if respuesta == "n":
            return False
        print("Ingresa S para Sí o N para No.")

def leer_prioridad():
    opciones = {"1": "emergencia", "2": "urgente", "3": "leve"}
    while True:
        entrada = input("Prioridad => [1] Emergencia | [2] Urgente | [3] Leve : ").strip()
        if entrada in opciones:
            return opciones[entrada]
        print("Ingresa un número del 1 al 3.")

def leer_especialidad():
    print("Especialidad => [1] Cardiología [2] Pediatría [3] Traumatología [4] Medicina general : ")
    opciones = {"1": "cardiologia", "2": "pediatria", "3": "traumatologia", "4": "medicina general"}
    while True:
        entrada = input("Selecciona: ").strip()
        if entrada in opciones:
            return opciones[entrada]
        print("Opción inválida.")

def menu():
    hospital = Hospital(camas=8, diagnosticos=7, cantidad_consultorios=10)
    while True:
        print("\n" + "SISTENA DE COLA DE ATENCION HOSPITALARIA".center(80, "*") )
        print(f"| Médicos: {hospital.total_medicos} / Consultorios: {len(hospital.consultorios)} / Camas: {hospital.total_camas} / Diagnósticos: {hospital.total_diagnosticos}\n|")
        print("| [1] Registrar paciente\n| [2] Registrar médico\n| [3] Ejecutar simulación y comparar algoritmos\n| [4] Mostrar médicos, turnos y consultorios")
        print("| [5] Reiniciar simulación\n| [0] salir\n|")
        print("| Presione [D] para cargar datos de demostración y ejecutar")
        print("*"*80)
        opcion = input("Elige una opción: ").strip().lower()

        if opcion == "1":
            if hospital.ejecutado:
                print("La simulación ya fue procesada. Reiníciala antes de registrar más pacientes.")
            else:
                nombre = input("Nombre del paciente: ").strip()
                if not nombre:
                    print("El nombre no puede estar vacío.")
                    continue
                while True:
                    hora_llegada = input("Hora de llegada del paciente (HH:MM, formato 24 horas): ").strip()
                    try:
                        llegada = hospital.convertir_hora_llegada(hora_llegada)
                        break
                    except ValueError as error:
                        print("Hora inválida:", error)
                #llegada = leer_numero(f"Minutos después del inicio de jornada ({hospital.hora_actual(0)}): ")
                duracion = leer_numero("Duración estimada de atención (minutos): ", 1)
                prioridad = leer_prioridad()
                especialidad = leer_especialidad()
                cama = leer_booleano("¿Necesita cama? (S/N): ")
                diagnostico = leer_booleano("¿Necesita diagnóstico? (S/N): ")
                try:
                    hospital.registrar_paciente(nombre, llegada, duracion, prioridad, especialidad, cama, diagnostico)
                except ValueError as error:
                    print("Error:", error)

        elif opcion == "2":
            if hospital.ejecutado:
                print("Reinicia la simulación antes de modificar la plantilla médica.")
                continue
            nombre_medico = input("Nombre del médico: ").strip()
            especialidad_medico = leer_especialidad()
            hora_inicio = input("Hora de inicio (HH:MM): ").strip()
            hora_fin = input("Hora de fin (HH:MM): ").strip()
            try:
                hospital.registrar_medico(nombre_medico, especialidad_medico, hora_inicio, hora_fin)
            except ValueError as error:
                print("Error:", error)

        elif opcion == "3":
            if not hospital.pacientes:
                print("No hay pacientes registrados.")
                continue
            if hospital.ejecutado:
                print("Reinicia la simulación antes de ejecutarla nuevamente.")
                continue
            #comparación inicial con todos los pacientes que llegan al minuto cero
            candidatos = [p for p in hospital.pacientes if p.Llegada == 0]
            disponibles, mapa = hospital.medicos_elegibles(candidatos)
            comparar_algoritmos(candidatos, mapa, hospital.total_camas, hospital.total_diagnosticos,min(len(hospital.consultorios), len(disponibles)), 0)
            print("\n" + ("PROGRAMACIÓN DINÁMICA" if opcion == "2" else "DIVIDE Y VENCERÁS").center(100, "/"))
            hospital.simular(1 if opcion == "2" else 2)

        elif opcion == "4":
            hospital.mostrar_medicos()
            print("\nCONSULTORIOS:")
            for c in hospital.consultorios:
                estado = f"ocupado por médico #{c.ocupado_por}" if c.ocupado_por else "libre"
                print(f"Consultorio {c.Id}: {estado}")

        elif opcion == "5":
            hospital.reiniciar()

        elif opcion == "d" or opcion == "D":
            hospitalDemostracion = Hospital(camas=5, diagnosticos=3, cantidad_consultorios=4)
            print(f"\n{"DEMOSTRACION INICIADA".center(120,"*")}\n")
            datos_demostracion(hospitalDemostracion)
            disponibles, mapa = hospitalDemostracion.medicos_elegibles(hospitalDemostracion.pacientes)
            comparar_algoritmos(hospitalDemostracion.pacientes, mapa, hospitalDemostracion.total_camas, hospitalDemostracion.total_diagnosticos,
                                min(len(hospitalDemostracion.consultorios), len(disponibles)), 0)
            print("\n" + "DEMOSTRACIÓN CON PROGRAMACIÓN DINÁMICA".center(100, "/"))
            hospitalDemostracion.simular(1)
            hospitalDemostracion.reiniciar()
            print("\n" + "DEMOSTRACIÓN CON DIVIDE Y VENCERÁS".center(100, "/"))
            hospitalDemostracion.simular(2)
            print(f"\n{"DEMOSTRACION FINALIZADA".center(120,"*")}\n")

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